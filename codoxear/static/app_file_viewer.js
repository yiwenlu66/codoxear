(function () {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file viewer dependency missing: ${name}`);
    return value;
  }

  function requireStatusNode(value) {
    if (!value || typeof value.replaceChildren !== "function") throw new TypeError("file viewer dependency missing: fileStatus");
    return value;
  }

  function requireEditButtonNode(value) {
    if (!value || !value.classList || typeof value.classList.toggle !== "function" || typeof value.setAttribute !== "function") {
      throw new TypeError("file viewer dependency missing: fileEditButton");
    }
    return value;
  }

  const BROWSER_SAFE_VIDEO_TYPES = new Set(["video/mp4", "video/webm", "video/ogg"]);

  function bindFileTouchPress(button, handler, options = {}) {
    if (!button || typeof button.addEventListener !== "function" || typeof handler !== "function") return false;
    const nowMs = typeof options.nowMs === "function" ? options.nowMs : () => Date.now();
    let suppressClickUntil = 0;
    let sawPointerTouchAt = 0;
    const run = (event) => {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      suppressClickUntil = nowMs() + 700;
      handler();
    };
    button.addEventListener("pointerdown", (event) => {
      if (event && event.pointerType === "touch") sawPointerTouchAt = nowMs();
      run(event);
    });
    button.addEventListener(
      "touchstart",
      (event) => {
        if (nowMs() - sawPointerTouchAt < 700) {
          event.preventDefault();
          event.stopPropagation();
          return;
        }
        run(event);
      },
      { passive: false }
    );
    button.addEventListener("click", (event) => {
      if (nowMs() < suppressClickUntil) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      run(event);
    });
    return true;
  }

  function bindFileTouchClick(button, handler) {
    if (!button || typeof button.addEventListener !== "function" || typeof handler !== "function") return false;
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      handler();
    });
    return true;
  }

  function requireStyledNode(value, name) {
    if (!value || !value.style) throw new TypeError(`file viewer dependency missing: ${name}`);
    return value;
  }

  function requirePasteInput(value) {
    if (!value || !("value" in value)) throw new TypeError("file viewer dependency missing: filePasteInput");
    return value;
  }

  function createFilePasteDialogRuntime(options = {}) {
    const backdrop = requireStyledNode(options.backdrop, "filePasteBackdrop");
    const dialog = requireStyledNode(options.dialog, "filePasteDialog");
    const input = requirePasteInput(options.input);
    const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
    const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
    const focusActiveEditor = requireFunction(options.focusActiveEditor, "focusActiveEditor");
    const requestFrame = requireFunction(options.requestAnimationFrame, "requestAnimationFrame");

    function isOpen() {
      return dialog.style.display === "flex";
    }

    function hide({ restoreFocus = false } = {}) {
      backdrop.style.display = "none";
      dialog.style.display = "none";
      input.value = "";
      afterModalVisibilityChanged();
      if (restoreFocus) focusActiveEditor();
      return true;
    }

    function show() {
      prepareModalOpen();
      input.value = "";
      backdrop.style.display = "block";
      dialog.style.display = "flex";
      afterModalVisibilityChanged();
      requestFrame(() => {
        if (!isOpen()) return;
        try {
          input.focus({ preventScroll: true });
          input.select();
        } catch (_) {}
      });
      return true;
    }

    return Object.freeze({ hide, isOpen, show });
  }

  function timeoutPromise(promise, timeoutMs, message) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(message)), timeoutMs);
      promise.then(
        (value) => {
          clearTimeout(timer);
          resolve(value);
        },
        (error) => {
          clearTimeout(timer);
          reject(error);
        }
      );
    });
  }

  function createPdfLoader(options = {}) {
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const globalObject = options.globalObject || window;
    const timeoutMs = Math.max(1, Number(options.timeoutMs || 6000));
    const importModule = typeof options.importModule === "function" ? options.importModule : (url) => import(url);
    let readyPromise = null;

    function ensure() {
      if (readyPromise) return readyPromise;
      if (globalObject.pdfjsLib && typeof globalObject.pdfjsLib.getDocument === "function") {
        readyPromise = Promise.resolve(globalObject.pdfjsLib);
      } else {
        readyPromise = timeoutPromise(importModule(resolveAppUrl("pdf.mjs")), timeoutMs, "PDF renderer timed out");
      }
      readyPromise = readyPromise.then((pdfjs) => {
        if (pdfjs && pdfjs.GlobalWorkerOptions) pdfjs.GlobalWorkerOptions.workerSrc = resolveAppUrl("pdf.worker.mjs");
        return pdfjs;
      });
      readyPromise.catch(() => {
        readyPromise = null;
      });
      return readyPromise;
    }

    return Object.freeze({ ensure });
  }

  function fileSaveConflictTarget(sessionId, path) {
    return Object.freeze({ sessionId, path });
  }

  function createFileViewerController(deps) {
    const el = requireFunction(deps && deps.el, "el");
    const fileStatus = requireStatusNode(deps && deps.fileStatus);
    const fileEditButton = requireEditButtonNode(deps && deps.fileEditButton);
    const iconSvg = requireFunction(deps && deps.iconSvg, "iconSvg");
    const currentSessionId = requireFunction(deps && deps.currentSessionId, "currentSessionId");
    const currentFileSessionId = requireFunction(deps && deps.currentFileSessionId, "currentFileSessionId");
    const normalizeLineNumber = requireFunction(deps && deps.normalizeLineNumber, "normalizeLineNumber");
    const normalizeFileApiPath = requireFunction(deps && deps.normalizeFileApiPath, "normalizeFileApiPath");
    const isFileViewerOpen = requireFunction(deps && deps.isFileViewerOpen, "isFileViewerOpen");
    const hideFileUnsavedDialog = requireFunction(deps && deps.hideFileUnsavedDialog, "hideFileUnsavedDialog");
    const resetFileSearchState = requireFunction(deps && deps.resetFileSearchState, "resetFileSearchState");
    const closeFilePickerMenu = requireFunction(deps && deps.closeFilePickerMenu, "closeFilePickerMenu");
    const isTextFileKind = requireFunction(deps && deps.isTextFileKind, "isTextFileKind");
    const isDiffableFileKind = requireFunction(deps && deps.isDiffableFileKind, "isDiffableFileKind");
    const confirmReload = requireFunction(deps && deps.confirmReload, "confirmReload");
    const promptUnsavedFileChoice = requireFunction(deps && deps.promptUnsavedFileChoice, "promptUnsavedFileChoice");
    const restoreFileEditorText = requireFunction(deps && deps.restoreFileEditorText, "restoreFileEditorText");
    const hideFileViewer = requireFunction(deps && deps.hideFileViewer, "hideFileViewer");
    const setFilePath = requireFunction(deps && deps.setFilePath, "setFilePath");
    const resetFileViewerPanel = requireFunction(deps && deps.resetFileViewerPanel, "resetFileViewerPanel");
    const applyFileLoadResult = requireFunction(deps && deps.applyFileLoadResult, "applyFileLoadResult");
    const normalizeDraftFilePath = requireFunction(deps && deps.normalizeDraftFilePath, "normalizeDraftFilePath");
    const inspectSessionFilePath = requireFunction(deps && deps.inspectSessionFilePath, "inspectSessionFilePath");
    const api = requireFunction(deps && deps.api, "api");
    const focusEditor = requireFunction(deps && deps.focusEditor, "focusEditor");
    const disposeOpenRender = requireFunction(deps && deps.disposeOpenRender, "disposeOpenRender");
    const persistFileViewMode = requireFunction(deps && deps.persistFileViewMode, "persistFileViewMode");
    const persistFileNonDiffMode = requireFunction(deps && deps.persistFileNonDiffMode, "persistFileNonDiffMode");
    const isMarkdownPreviewable = requireFunction(deps && deps.isMarkdownPreviewable, "isMarkdownPreviewable");
    const updateFileTouchToolbar = requireFunction(deps && deps.updateFileTouchToolbar, "updateFileTouchToolbar");
    const useTouchFileEditorControls = requireFunction(deps && deps.useTouchFileEditorControls, "useTouchFileEditorControls");
    const hasActiveFileCodeEditor = requireFunction(deps && deps.hasActiveFileCodeEditor, "hasActiveFileCodeEditor");
    const hasBlockingFileEditorModal = requireFunction(deps && deps.hasBlockingFileEditorModal, "hasBlockingFileEditorModal");
    const isTextEntryTarget = requireFunction(deps && deps.isTextEntryTarget, "isTextEntryTarget");
    const eventTargetElement = requireFunction(deps && deps.eventTargetElement, "eventTargetElement");
    const normalizeFileEditorPosition = requireFunction(deps && deps.normalizeFileEditorPosition, "normalizeFileEditorPosition");
    const applyFileEditorSelection = requireFunction(deps && deps.applyFileEditorSelection, "applyFileEditorSelection");
    const isCollapsedFileSelection = requireFunction(deps && deps.isCollapsedFileSelection, "isCollapsedFileSelection");
    const fileHelpers = window.CodoxearFileHelpers || {};
    const positionAfterInsertedText =
      typeof (deps && deps.positionAfterInsertedText) === "function"
        ? deps.positionAfterInsertedText
        : requireFunction(fileHelpers.positionAfterInsertedText, "CodoxearFileHelpers.positionAfterInsertedText");
    const fileEditorEditSupportAvailable = requireFunction(deps && deps.fileEditorEditSupportAvailable, "fileEditorEditSupportAvailable");
    const updateFileDiffEditorOptions = requireFunction(deps && deps.updateFileDiffEditorOptions, "updateFileDiffEditorOptions");
    const showFilePasteDialog = requireFunction(deps && deps.showFilePasteDialog, "showFilePasteDialog");
    const hideFilePasteDialog = requireFunction(deps && deps.hideFilePasteDialog, "hideFilePasteDialog");
    const clipboardReadAvailable = requireFunction(deps && deps.clipboardReadAvailable, "clipboardReadAvailable");
    const readClipboardText = requireFunction(deps && deps.readClipboardText, "readClipboardText");
    const fileEditorDeleteCommandForKey =
      typeof (deps && deps.fileEditorDeleteCommandForKey) === "function"
        ? deps.fileEditorDeleteCommandForKey
        : requireFunction(fileHelpers.fileEditorDeleteCommandForKey, "CodoxearFileHelpers.fileEditorDeleteCommandForKey");
    const isActiveFileEditorInput = requireFunction(deps && deps.isActiveFileEditorInput, "isActiveFileEditorInput");
    const getActiveFileSelectionText = requireFunction(deps && deps.getActiveFileSelectionText, "getActiveFileSelectionText");
    const copyToClipboard = requireFunction(deps && deps.copyToClipboard, "copyToClipboard");
    const focusActiveFileCodeEditor = requireFunction(deps && deps.focusActiveFileCodeEditor, "focusActiveFileCodeEditor");
    const nowMs = requireFunction(deps && deps.nowMs, "nowMs");
    const setToast = requireFunction(deps && deps.setToast, "setToast");
    const renderMonacoFile = requireFunction(deps && deps.renderMonacoFile, "renderMonacoFile");
    const getFileEditorText = requireFunction(deps && deps.getFileEditorText, "getFileEditorText");
    const fmtBytes = requireFunction(deps && deps.fmtBytes, "fmtBytes");
    const applyFileMode = requireFunction(deps && deps.applyFileMode, "applyFileMode");
    const rememberOpenedFile = requireFunction(deps && deps.rememberOpenedFile, "rememberOpenedFile");
    const historyFileSelectionForSession = requireFunction(deps && deps.historyFileSelectionForSession, "historyFileSelectionForSession");
    const renderFilePickerMenu = requireFunction(deps && deps.renderFilePickerMenu, "renderFilePickerMenu");
    let activeSaveConflict = null;
    let fileOpenRequestId = 0;
    let fileOpenAbortController = null;
    let fileViewerSessionId = "";
    let fileViewerSessionSyncToken = 0;
    let fileCandidateRequestSeq = 0;
    let fileCandidateList = [];
    let fileEntryMap = new Map();
    let fileCandidateGitStateFresh = false;
    let fileCandidateCache = new Map();
    let fileSaveSeq = 0;
    let activeFileSaveToken = 0;
    let fileSavePending = false;
    let fileDirty = false;
    let fileEditMode = false;
    let fileEditorKind = "";
    let fileEditorProgrammaticChange = false;
    let fileUnsavedPromptResolver = null;
    let fileViewMode = normalizeFileViewMode(deps && deps.initialFileViewMode);
    let fileNonDiffMode = deps && deps.initialFileNonDiffMode === "preview" ? "preview" : "file";
    let activeFilePath = "";
    let activeFileApiPath = "";
    let activeFileGitPath = false;
    let activeFileLine = null;
    let activeFileKind = "";
    let activeFileText = "";
    let activeFileEditable = false;
    let activeFileVersion = "";
    let activeFileDraft = false;
    let activeVideoFallback = null;
    let activePdfRender = null;
    let unavailableSessionId = "";
    let fileSessionSelections = new Map();
    let fileTouchSelectMode = false;
    let fileTouchSelectAnchor = null;
    let fileTouchSelectHead = null;
    let fileTouchSelectGoalColumn = null;
    let fileTouchDeleteNativeSuppressUntil = 0;
    let fileViewerReturnFocusElement = null;
    let fileUnsavedReturnFocusElement = null;

    function focusReturnElement(value, ElementCtor = null) {
      const Ctor = typeof ElementCtor === "function" ? ElementCtor : null;
      if (!value || (Ctor && !(value instanceof Ctor))) return null;
      return value;
    }

    function normalizeFileViewMode(mode) {
      return mode === "preview" ? "preview" : mode === "file" ? "file" : "diff";
    }

    function currentFileViewMode() {
      return fileViewMode;
    }

    function setFileViewerReturnFocusElement(value, ElementCtor = null) {
      fileViewerReturnFocusElement = focusReturnElement(value, ElementCtor);
      return fileViewerReturnFocusElement;
    }

    function takeFileViewerReturnFocusElement() {
      const value = fileViewerReturnFocusElement;
      fileViewerReturnFocusElement = null;
      return value;
    }

    function setFileUnsavedReturnFocusElement(value, ElementCtor = null) {
      fileUnsavedReturnFocusElement = focusReturnElement(value, ElementCtor);
      return fileUnsavedReturnFocusElement;
    }

    function takeFileUnsavedReturnFocusElement() {
      const value = fileUnsavedReturnFocusElement;
      fileUnsavedReturnFocusElement = null;
      return value;
    }

    function currentFileNonDiffMode() {
      return fileNonDiffMode;
    }

    function setFileViewMode(mode) {
      const next = normalizeFileViewMode(mode);
      fileViewMode = next;
      persistFileViewMode(fileViewMode);
      if (next !== "diff") {
        fileNonDiffMode = next;
        persistFileNonDiffMode(fileNonDiffMode);
      }
      applyFileMode();
    }

    function normalizeSessionId(value) {
      return String(value || "").trim();
    }

    function isFileViewerSessionUnavailable() {
      const sid = normalizeSessionId(currentSessionId());
      return Boolean(unavailableSessionId && sid && unavailableSessionId === sid);
    }

    function isUnavailable() {
      return isFileViewerSessionUnavailable();
    }

    function clearFileViewerUnavailableSession() {
      unavailableSessionId = "";
    }

    function disableFileViewerForUnavailableSession(sessionId) {
      const sid = normalizeSessionId(sessionId);
      if (!sid) return false;
      rememberActiveFileSelection(sid);
      invalidateFileViewerSessionSync();
      unavailableSessionId = sid;
      clearActiveFileSaveState();
      setFileEditMode(false);
      hideFileUnsavedDialog("cancel");
      cancelPendingFileOpen();
      resetFileSearchState();
      closeFilePickerMenu({ restoreInput: true });
      syncFileEditorReadOnly();
      fileStatus.textContent = "Session is no longer available; copy unsaved edits before closing.";
      updateFileEditButton();
      updateFileTouchToolbar();
      return true;
    }

    function handleFileViewerSessionUnavailable(sessionId) {
      const sid = normalizeSessionId(sessionId);
      if (!sid || !isFileViewerOpen()) return false;
      const viewerSessionId = normalizeSessionId(currentSessionId());
      if (viewerSessionId && viewerSessionId !== sid) return false;
      if (!currentFileDirty()) {
        hideFileViewer();
        return true;
      }
      return disableFileViewerForUnavailableSession(sid);
    }

    function nextActiveFileIdentity(current, nextPath, { gitPath = undefined, apiPath = undefined } = {}) {
      if (!current || typeof current !== "object") throw new Error("current file identity required");
      const previousPath = String(current.path ?? "");
      const previousApiPath = String(current.apiPath || "");
      const rel = String(nextPath ?? "");
      const useGitPath = gitPath === undefined ? Boolean(current.gitPath) : Boolean(gitPath);
      const reusableApiPath = rel === previousPath ? previousApiPath : "";
      return Object.freeze({
        path: rel,
        gitPath: useGitPath,
        apiPath: apiPath === undefined ? (useGitPath ? fileApiPathForPath(rel, reusableApiPath) : "") : normalizeFileApiPath(apiPath),
      });
    }

    function currentActiveFileIdentity() {
      return Object.freeze({ path: String(activeFilePath ?? ""), gitPath: Boolean(activeFileGitPath), apiPath: String(activeFileApiPath || "") });
    }

    function rememberActiveFileSelection(sessionId = currentFileSessionId()) {
      const sid = String(sessionId || "").trim();
      const identity = currentActiveFileIdentity();
      const path = String(identity.path ?? "");
      if (!sid || path === "") return;
      const line = currentActiveFileLine();
      fileSessionSelections.set(sid, {
        path,
        apiPath: identity.apiPath || "",
        line: line == null ? null : line,
        gitPath: Boolean(identity.gitPath),
      });
    }

    function preferredFileSelectionForSession(sessionId) {
      const sid = String(sessionId || "").trim();
      if (!sid) return { path: "", line: null, gitPath: false };
      const remembered = fileSessionSelections.get(sid);
      const rememberedPath = remembered && typeof remembered.path === "string" ? remembered.path : "";
      if (rememberedPath !== "") {
        return {
          path: rememberedPath,
          apiPath: normalizeFileApiPath(remembered.apiPath),
          line: normalizeLineNumber(remembered.line),
          gitPath: Boolean(remembered.gitPath),
        };
      }
      return historyFileSelectionForSession(sid);
    }

    function currentActiveFileLine() {
      return activeFileLine;
    }

    function fileCandidateKey(path, gitPath = false, apiPath = "") {
      const identity = gitPath && apiPath ? normalizeFileApiPath(apiPath) : String(path ?? "");
      return `${gitPath ? "git" : "session"}\u0000${identity}`;
    }

    function normalizeFileCandidateSource(source) {
      const value = String(source || "").trim();
      if (value === "changed" || value === "mentioned" || value === "recent") return value;
      return "";
    }

    function cloneFileCandidateEntry(entry) {
      if (!entry || typeof entry.path !== "string" || entry.path === "") return null;
      const source = normalizeFileCandidateSource(entry.source);
      const gitPath = entry.gitPath === undefined ? Boolean(entry.changed && source === "changed") : Boolean(entry.gitPath);
      const apiPath = normalizeFileApiPath(entry.apiPath || entry.api_path);
      return {
        path: entry.path,
        apiPath,
        gitPath,
        key: fileCandidateKey(entry.path, gitPath, apiPath),
        additions: entry.additions ?? null,
        deletions: entry.deletions ?? null,
        changed: Boolean(entry.changed),
        source,
      };
    }

    function fileCandidateKeyForEntry(entry) {
      return fileCandidateKey(entry && entry.path, Boolean(entry && entry.gitPath), normalizeFileApiPath(entry && entry.apiPath));
    }

    function applyFileCandidateEntries(entries) {
      const nextList = [];
      const nextMap = new Map();
      for (const raw of Array.isArray(entries) ? entries : []) {
        const entry = cloneFileCandidateEntry(raw);
        if (!entry || nextMap.has(entry.key)) continue;
        nextList.push(entry.key);
        nextMap.set(entry.key, entry);
      }
      fileCandidateList = nextList;
      fileEntryMap = nextMap;
    }

    function currentFileCandidateKeys() {
      return fileCandidateList.slice();
    }

    function currentFileCandidateEntries() {
      return fileCandidateList
        .map((key) => cloneFileCandidateEntry(fileEntryMap.get(key)))
        .filter(Boolean);
    }

    function fileEntryForKey(key) {
      return cloneFileCandidateEntry(fileEntryMap.get(String(key || "")));
    }

    function fileEntryForPath(path, gitPath = false, apiPath = "") {
      const token = normalizeFileApiPath(apiPath);
      const preferred = fileEntryMap.get(fileCandidateKey(path, gitPath, token));
      if (preferred) return cloneFileCandidateEntry(preferred);
      const fallback = fileEntryMap.get(fileCandidateKey(path, gitPath));
      if (fallback && (!token || !fallback.apiPath || fallback.apiPath === token)) return cloneFileCandidateEntry(fallback);
      for (const key of fileCandidateList) {
        const entry = fileEntryMap.get(key);
        if (!entry || entry.path !== path || Boolean(entry.gitPath) !== Boolean(gitPath)) continue;
        if (!token || normalizeFileApiPath(entry.apiPath) === token) return cloneFileCandidateEntry(entry);
      }
      return null;
    }

    function fileApiPathForPath(path, apiPath = "") {
      const existing = normalizeFileApiPath(apiPath);
      if (existing) return existing;
      const entry = fileEntryForPath(path, true);
      return normalizeFileApiPath(entry && entry.apiPath);
    }

    function activeFileEntry() {
      const identity = currentActiveFileIdentity();
      if (!identity.path) return null;
      return fileEntryForPath(identity.path, identity.gitPath, identity.apiPath);
    }

    function isGitFileCandidatePath(path, changed = null, gitPath = null, apiPath = "") {
      if (gitPath !== null && gitPath !== undefined) return Boolean(gitPath);
      if (changed !== null && changed !== undefined) return Boolean(changed);
      const gitEntry = fileEntryForPath(path, true, normalizeFileApiPath(apiPath));
      if (gitEntry) return true;
      const sessionEntry = fileEntryForPath(path, false);
      return Boolean(sessionEntry && sessionEntry.gitPath);
    }

    function currentFileCandidateGitStateFresh() {
      return fileCandidateGitStateFresh;
    }

    function setFileCandidateGitStateFresh(fresh) {
      fileCandidateGitStateFresh = Boolean(fresh);
      return fileCandidateGitStateFresh;
    }

    function rememberFileCandidateCache(sessionId, key, now = Date.now()) {
      const sid = String(sessionId || "").trim();
      if (!sid || !key) return false;
      fileCandidateCache.set(sid, { key, ts: Number(now || 0), entries: currentFileCandidateEntries() });
      return true;
    }

    function fileCandidateCacheEntry(sessionId) {
      const sid = String(sessionId || "").trim();
      const cached = sid ? fileCandidateCache.get(sid) : null;
      if (!cached || typeof cached !== "object") return null;
      return Object.freeze({ key: String(cached.key || ""), ts: Number(cached.ts || 0), entries: Array.isArray(cached.entries) ? cached.entries.map(cloneFileCandidateEntry).filter(Boolean) : [] });
    }

    function deleteFileCandidateCache(sessionId) {
      const sid = String(sessionId || "").trim();
      if (!sid) return false;
      return fileCandidateCache.delete(sid);
    }

    function fileCandidateCacheSize() {
      return fileCandidateCache.size;
    }

    function applyFileCandidateRefreshEntries(entries, { gitStateFresh = false } = {}) {
      applyFileCandidateEntries(entries);
      setFileCandidateGitStateFresh(gitStateFresh);
      applyFileMode();
      return true;
    }

    function clearFileCandidateRefreshEntries() {
      return applyFileCandidateRefreshEntries([], { gitStateFresh: false });
    }

    function applyFreshFileCandidateCache(sid, key, { now = Date.now(), ttl = 0 } = {}) {
      const cached = fileCandidateCacheEntry(sid);
      if (!cached || cached.key !== key) return false;
      const age = Number(now || 0) - Number(cached.ts || 0);
      if (!(age >= 0 && age < Number(ttl || 0))) return false;
      return applyFileCandidateRefreshEntries(cached.entries, { gitStateFresh: false });
    }

    function upsertFileEntry(entry) {
      const merged = cloneFileCandidateEntry(entry);
      if (!merged) return false;
      const current = fileEntryMap.get(merged.key);
      const next = current && !merged.source ? { ...merged, source: normalizeFileCandidateSource(current.source) } : merged;
      if (!fileEntryMap.has(next.key)) fileCandidateList.push(next.key);
      fileEntryMap.set(next.key, Object.freeze({ ...next }));
      return true;
    }

    function pickerEntryForKey(key, { score = 0 } = {}) {
      const entry = fileEntryForKey(key);
      return entry ? { ...entry, added: true, score } : null;
    }

    function pickerEntryForPath(path, { score = 0, gitPath = false } = {}) {
      const key = fileCandidateKey(path, gitPath);
      const entry = cloneFileCandidateEntry(fileEntryMap.get(key) || { path, gitPath, additions: null, deletions: null, changed: false, source: "" });
      if (!entry) return null;
      return { ...entry, added: fileEntryMap.has(key), score };
    }

    function resolveFileViewerOpenTarget({ sessionId = "", explicitPath = "", explicitLine = null } = {}) {
      const sid = String(sessionId || "").trim();
      if (!sid) return Object.freeze({ kind: "none" });
      const requestedPath = String(explicitPath ?? "");
      if (requestedPath) {
        return Object.freeze({ kind: "path", source: "explicit", path: requestedPath, line: normalizeLineNumber(explicitLine), changed: null, gitPath: false, apiPath: "" });
      }
      const preferred = preferredFileSelectionForSession(sid);
      if (preferred.path) {
        return Object.freeze({ kind: "path", source: "preferred", path: preferred.path, line: preferred.line, changed: null, gitPath: Boolean(preferred.gitPath), apiPath: normalizeFileApiPath(preferred.apiPath) });
      }
      const firstKey = fileCandidateList.length ? fileCandidateList[0] : "";
      const first = firstKey ? fileEntryForKey(firstKey) : null;
      if (first) {
        return Object.freeze({ kind: "path", source: "first", path: first.path, line: null, changed: Boolean(first.changed), gitPath: Boolean(first.gitPath), apiPath: normalizeFileApiPath(first.apiPath) });
      }
      return Object.freeze({ kind: "none" });
    }

    function currentFileViewerSessionId() {
      return normalizeSessionId(fileViewerSessionId);
    }

    function setFileViewerSessionId(sessionId) {
      fileViewerSessionId = normalizeSessionId(sessionId);
      return fileViewerSessionId;
    }

    function clearFileViewerSessionId() {
      fileViewerSessionId = "";
    }

    function beginFileViewerSessionSync() {
      fileViewerSessionSyncToken += 1;
      return fileViewerSessionSyncToken;
    }

    function invalidateFileViewerSessionSync() {
      fileViewerSessionSyncToken += 1;
      return fileViewerSessionSyncToken;
    }

    function isCurrentFileViewerSessionSync(token) {
      return token === fileViewerSessionSyncToken;
    }

    function beginFileCandidateRefresh() {
      fileCandidateRequestSeq += 1;
      return fileCandidateRequestSeq;
    }

    function isCurrentFileCandidateRefresh(requestSeq) {
      return requestSeq === fileCandidateRequestSeq;
    }

    function currentFileEditMode() {
      return fileEditMode;
    }

    function normalizeFileEditorKind(kind) {
      const nextKind = String(kind || "");
      if (nextKind !== "" && nextKind !== "file" && nextKind !== "diff" && nextKind !== "plain-fallback") throw new Error("invalid file editor kind");
      return nextKind;
    }

    function currentFileEditorKind() {
      return fileEditorKind;
    }

    function isFileEditorProgrammaticChange() {
      return fileEditorProgrammaticChange;
    }

    function beginFileEditorProgrammaticChange() {
      fileEditorProgrammaticChange = true;
      return true;
    }

    function finishFileEditorProgrammaticChange() {
      fileEditorProgrammaticChange = false;
      return true;
    }

    function runFileEditorProgrammaticChange(callback) {
      const fn = requireFunction(callback, "runFileEditorProgrammaticChange");
      beginFileEditorProgrammaticChange();
      try {
        return fn();
      } finally {
        finishFileEditorProgrammaticChange();
      }
    }

    function setFileEditorKind(kind) {
      fileEditorKind = normalizeFileEditorKind(kind);
      return fileEditorKind;
    }

    function setFileEditMode(nextMode) {
      fileEditMode = Boolean(nextMode) && activeFileEditModeAllowedInCurrentView();
      syncFileEditorReadOnly();
      updateFileEditButton();
    }

    function currentActiveFileKind() {
      return activeFileKind;
    }

    function currentActiveFileText() {
      return activeFileText;
    }

    function currentActiveFileEditable() {
      return activeFileEditable;
    }

    function currentActiveFileVersion() {
      return activeFileVersion;
    }

    function currentActiveFileDraft() {
      return activeFileDraft;
    }

    function resetActiveFileBufferState() {
      activeFileKind = "";
      activeFileText = "";
      activeFileEditable = false;
      activeFileVersion = "";
      activeFileDraft = false;
      fileEditMode = false;
      clearActiveFileSaveState();
      resetFileTouchSelectionState();
      fileDirty = false;
      updateFileEditButton();
    }

    function applyActiveFileTextState({ kind = "text", text = "", editable = false, version = "", draft = false } = {}) {
      const nextKind = String(kind || "text");
      if (nextKind !== "text" && nextKind !== "markdown") throw new Error("invalid active file text kind");
      activeFileKind = nextKind;
      activeFileText = String(text ?? "");
      activeFileEditable = Boolean(editable);
      activeFileVersion = typeof version === "string" ? version : "";
      activeFileDraft = Boolean(draft);
    }

    function applyActiveFileDiffState({ currentText = "", currentExists = false } = {}) {
      applyActiveFileTextState({ kind: "text", text: currentText, editable: Boolean(currentExists), version: "", draft: false });
    }

    function applyActiveFileNonTextState(kind) {
      const nextKind = String(kind || "");
      if (nextKind !== "image" && nextKind !== "pdf" && nextKind !== "video" && nextKind !== "download_only") throw new Error("invalid active file non-text kind");
      activeFileKind = nextKind;
      activeFileText = "";
      activeFileEditable = false;
      activeFileVersion = "";
      activeFileDraft = false;
    }

    function clearActiveFileIdentity({ line = null } = {}) {
      activeFilePath = "";
      activeFileApiPath = "";
      activeFileGitPath = false;
      activeFileLine = normalizeLineNumber(line);
    }

    function setActiveFileIdentity(nextPath, { line = null, gitPath = undefined, apiPath = undefined } = {}) {
      const identity = nextActiveFileIdentity(currentActiveFileIdentity(), nextPath, { gitPath, apiPath });
      activeFilePath = identity.path;
      activeFileGitPath = identity.gitPath;
      activeFileApiPath = identity.apiPath;
      activeFileLine = normalizeLineNumber(line);
      return Object.freeze({ ...identity, line: activeFileLine });
    }

    function beginActiveFileIdentity(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined } = {}) {
      const identity = nextActiveFileIdentity(currentActiveFileIdentity(), nextPath == null ? activeFilePath : nextPath, { gitPath, apiPath });
      activeFilePath = identity.path;
      activeFileGitPath = identity.gitPath;
      activeFileApiPath = identity.apiPath;
      activeFileLine = line === undefined ? activeFileLine : normalizeLineNumber(line);
      return Object.freeze({ ...identity, line: activeFileLine });
    }

    function abortPendingFileOpenTransport() {
      if (!fileOpenAbortController) return;
      try {
        fileOpenAbortController.abort();
      } catch (_) {}
      fileOpenAbortController = null;
    }

    function cancelPendingFileOpen() {
      fileOpenRequestId += 1;
      disposeOpenRender();
      abortPendingFileOpenTransport();
    }

    function beginFileOpenRequest(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined } = {}) {
      cancelPendingFileOpen();
      const identity = beginActiveFileIdentity(nextPath, { line, gitPath, apiPath });
      const controller = typeof AbortController === "function" ? new AbortController() : null;
      if (controller) fileOpenAbortController = controller;
      return Object.freeze({
        requestId: fileOpenRequestId,
        sessionId: currentSessionId(),
        path: identity.path,
        apiPath: identity.apiPath,
        gitPath: identity.gitPath,
        line: identity.line,
        signal: controller ? controller.signal : null,
      });
    }

    function isCurrentFileOpenRequest(request) {
      if (!request) return false;
      const identity = currentActiveFileIdentity();
      return Boolean(
        request.requestId === fileOpenRequestId &&
          request.sessionId === currentSessionId() &&
          request.path === String(identity.path ?? "") &&
          String(request.apiPath || "") === String(identity.apiPath || "")
      );
    }

    function finalizeFileOpenRequest(request) {
      if (!request || !fileOpenAbortController) return;
      if (fileOpenAbortController.signal !== request.signal) return;
      if (!isCurrentFileOpenRequest(request)) return;
      fileOpenAbortController = null;
    }

    function startFileOpenRequest(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined } = {}) {
      const request = beginFileOpenRequest(nextPath, { line, gitPath, apiPath });
      return Object.freeze({
        request,
        path: request.path,
        done: () => finalizeFileOpenRequest(request),
      });
    }

    function normalizeExplicitFileOpenMode(requestedMode) {
      if (requestedMode === null || requestedMode === undefined || requestedMode === "") return null;
      if (requestedMode === "preview" || requestedMode === "file" || requestedMode === "diff") return requestedMode;
      throw new Error("invalid file open mode");
    }

    function resolveFileOpenViewMode(request, rel, requestedMode = null) {
      const openMode = normalizeExplicitFileOpenMode(requestedMode);
      if (openMode) return openMode;
      const entry = activeFileEntry();
      const canUseDiffView = request && request.gitPath && currentFileCandidateGitStateFresh() && Boolean(entry && entry.changed);
      const viewMode = currentFileViewMode();
      return viewMode === "preview" && !isMarkdownPreviewable(rel) ? "file" : viewMode === "diff" && !canUseDiffView ? "file" : viewMode;
    }

    async function resolveFileOpenMode(path, { changed = null, gitPath = null, apiPath = "" } = {}) {
      const token = normalizeFileApiPath(apiPath);
      const useGitPath = gitPath === null || gitPath === undefined ? isGitFileCandidatePath(path, changed, null, token) : Boolean(gitPath);
      const identityEntry = fileEntryForPath(path, useGitPath, token);
      const requestApiPath = token || normalizeFileApiPath(identityEntry && identityEntry.apiPath);
      const candidateChanged = useGitPath && (changed === null || changed === undefined ? Boolean(identityEntry && identityEntry.changed) : Boolean(changed));
      const inspect = await inspectSessionFilePath(path, { gitPath: useGitPath, apiPath: requestApiPath });
      if (!inspect || !inspect.exists) {
        if (currentFileCandidateGitStateFresh() && candidateChanged) return "diff";
        throw new Error("file not found");
      }
      const kind = String(inspect.kind || "").trim();
      const isChanged = currentFileCandidateGitStateFresh() && candidateChanged;
      if (isChanged && isDiffableFileKind(kind)) return "diff";
      if (kind === "markdown" && currentFileNonDiffMode() === "preview") return "preview";
      return "file";
    }

    function isFileOpenAbortError(error) {
      return Boolean(error && error.name === "AbortError");
    }

    function blockUnavailableFileAction() {
      if (!isUnavailable()) return false;
      fileStatus.textContent = "Session is no longer available; copy unsaved edits before closing.";
      return true;
    }

    function currentFileEditorState() {
      const identity = currentActiveFileIdentity();
      return Object.freeze({
        path: String(identity.path || ""),
        apiPath: String(identity.apiPath || ""),
        gitPath: Boolean(identity.gitPath),
        kind: String(currentActiveFileKind() || ""),
        editable: Boolean(currentActiveFileEditable()),
        version: String(currentActiveFileVersion() || ""),
        draft: Boolean(currentActiveFileDraft()),
        viewMode: String(currentFileViewMode() || ""),
        editorKind: String(currentFileEditorKind() || ""),
        editMode: Boolean(currentFileEditMode()),
        dirty: Boolean(currentFileDirty()),
        savePending: isFileSavePending(),
        sessionId: String(currentSessionId() || ""),
        unavailable: isUnavailable(),
      });
    }

    function fileEditorCapabilities(state) {
      if (!state || typeof state !== "object") throw new Error("file editor state required");
      const kind = String(state.kind || "");
      const textKind = isTextFileKind(kind);
      const editable = Boolean(state.editable);
      const unavailable = Boolean(state.unavailable);
      const viewMode = String(state.viewMode || "");
      const editorKind = String(state.editorKind || "");
      const editMode = Boolean(state.editMode);
      const savePending = Boolean(state.savePending);
      const canEnterEditMode = Boolean(!unavailable && String(state.path || "") && !savePending && (!kind || textKind) && editorKind !== "plain-fallback" && editable);
      const writable = Boolean(editMode && editable && viewMode === "file" && !unavailable);
      const idleWritable = Boolean(writable && !savePending);
      const idleTextWritable = Boolean(idleWritable && textKind);
      const editModeAllowedInCurrentView = Boolean(viewMode === "file" && textKind && editable && !unavailable);
      return Object.freeze({ canEnterEditMode, writable, idleWritable, idleTextWritable, editModeAllowedInCurrentView });
    }

    function activeFileEditorCapabilities() {
      return fileEditorCapabilities(currentFileEditorState());
    }

    function activeFileCanEnterEditMode() {
      return activeFileEditorCapabilities().canEnterEditMode;
    }

    function activeFileEditorWritable() {
      return activeFileEditorCapabilities().writable;
    }

    function activeFileEditorIdleWritable() {
      return activeFileEditorCapabilities().idleWritable;
    }

    function activeFileEditorIdleTextWritable() {
      return activeFileEditorCapabilities().idleTextWritable;
    }

    function activeFileEditModeAllowedInCurrentView() {
      return activeFileEditorCapabilities().editModeAllowedInCurrentView;
    }

    function activeVideoFallbackSnapshot() {
      const state = activeVideoFallback;
      return state ? Object.freeze({ token: state.token, previewUrl: state.previewUrl, used: Boolean(state.used), preparing: Boolean(state.preparing), rel: state.rel, size: state.size }) : null;
    }

    function setActiveVideoFallback(nextState) {
      if (!nextState || !nextState.previewUrl) {
        activeVideoFallback = null;
        return null;
      }
      activeVideoFallback = {
        token: String(nextState.token || ""),
        previewUrl: String(nextState.previewUrl || ""),
        used: Boolean(nextState.used),
        preparing: Boolean(nextState.preparing),
        rel: String(nextState.rel || ""),
        size: typeof nextState.size === "number" ? nextState.size : 0,
      };
      return activeVideoFallbackSnapshot();
    }

    function clearActiveVideoFallback() {
      activeVideoFallback = null;
    }

    function currentActiveVideoFallback() {
      return activeVideoFallbackSnapshot();
    }

    function setActivePdfRenderState(state) {
      activePdfRender = state || null;
      return activePdfRender;
    }

    function takeActivePdfRenderState() {
      const state = activePdfRender;
      activePdfRender = null;
      return state;
    }

    function clearActivePdfRenderState() {
      activePdfRender = null;
      return true;
    }

    function isActivePdfRenderState(state) {
      return Boolean(state && activePdfRender === state);
    }

    function currentActiveVideoPreviewToken() {
      const state = activeVideoFallback;
      return state && state.token ? state.token : "";
    }

    function normalizedVideoContentType(value) {
      return typeof value === "string" ? value.split(";", 1)[0].trim().toLowerCase() : "";
    }

    function prepareActiveVideoLoadResult(rel, result, request) {
      applyActiveFileNonTextState("video");
      if (!result || typeof result.video_url !== "string" || !result.video_url) throw new Error("invalid video response");
      const path = String(rel || "video");
      const previewUrl = typeof result.video_preview_url === "string" ? result.video_preview_url : "";
      const size = typeof result.size === "number" ? result.size : 0;
      const contentType = normalizedVideoContentType(result.content_type);
      const token = `${request.requestId}:${path}:${nowMs()}`;
      const shouldPreviewFirst = Boolean(previewUrl && contentType && !BROWSER_SAFE_VIDEO_TYPES.has(contentType));
      setActiveVideoFallback(previewUrl ? { token, previewUrl, rel: path, size } : null);
      applyFileMode();
      return Object.freeze({
        token,
        rel: path,
        videoUrl: result.video_url,
        previewUrl,
        size,
        contentType,
        shouldPreviewFirst,
        initialStatus: `${path} - video - ${fmtBytes(size)}`,
      });
    }

    function handleActiveVideoLoadError(token, options = {}) {
      const clearVideoHandlers = requireFunction(options.clearVideoHandlers, "clearVideoHandlers");
      const loadPreview = requireFunction(options.loadPreview, "loadCompatibleVideoPreview");
      const expectedToken = String(token || "");
      const previewUrl = typeof options.previewUrl === "string" ? options.previewUrl : "";
      const fallback = activeVideoFallback;
      const rel = String((fallback && fallback.rel) || options.rel || "video");
      if (!fallback || fallback.token !== expectedToken) {
        if (!previewUrl) fileStatus.textContent = `${rel} - video unsupported`;
        return false;
      }
      if (clearUsedCompatibleVideoPreview(expectedToken)) {
        clearVideoHandlers();
        fileStatus.textContent = `${rel} - video preview unavailable after conversion`;
        return true;
      }
      void loadPreview(expectedToken, { explicit: false });
      return true;
    }

    function handleActiveVideoLoadedMetadata(token) {
      const expectedToken = String(token || "");
      const fallback = activeVideoFallback;
      if (!fallback || fallback.token !== expectedToken || !fallback.used) return false;
      fileStatus.textContent = `${fallback.rel || "video"} - compatible video preview - ${fmtBytes(fallback.size)}`;
      return true;
    }

    function prepareFileLoadResult(rel, result, request, { viewMode = "file" } = {}) {
      if (!isCurrentFileOpenRequest(request)) return null;
      if (!result || typeof result.kind !== "string") throw new Error("invalid response");
      const path = String(rel || "");
      if (result.kind === "diff") {
        const baseText = typeof result.baseText === "string" ? result.baseText : "";
        const currentText = typeof result.currentText === "string" ? result.currentText : "";
        applyActiveFileDiffState({ currentText, currentExists: result.currentExists });
        if (!result.baseExists && !result.currentExists) return Object.freeze({ kind: "diff", noDiff: true, status: `${path} - no diff` });
        return Object.freeze({ kind: "diff", noDiff: false, baseText, currentText, status: `${path} - diff` });
      }
      if (result.kind === "image") {
        applyActiveFileNonTextState("image");
        if (typeof result.image_url !== "string" || !result.image_url) throw new Error("invalid image response");
        const size = typeof result.size === "number" ? result.size : 0;
        return Object.freeze({ kind: "image", imageUrl: result.image_url, alt: path, status: `${path} - ${fmtBytes(size)}` });
      }
      if (result.kind === "pdf") {
        applyActiveFileNonTextState("pdf");
        if (typeof result.pdf_url !== "string" || !result.pdf_url) throw new Error("invalid pdf response");
        const size = typeof result.size === "number" ? result.size : 0;
        return Object.freeze({ kind: "pdf", pdfUrl: result.pdf_url, status: `${path} - PDF - ${fmtBytes(size)}` });
      }
      if (result.kind === "video") {
        return Object.freeze({ kind: "video", ...prepareActiveVideoLoadResult(path, result, request) });
      }
      if (result.kind === "download_only") {
        applyActiveFileNonTextState("download_only");
        const size = typeof result.size === "number" ? result.size : 0;
        return Object.freeze({ kind: "download_only", reason: String(result.reason || ""), viewerMaxBytes: Number(result.viewer_max_bytes || 0), size, status: `${path} - download only - ${fmtBytes(size)}` });
      }
      if (typeof result.text !== "string") throw new Error("invalid response");
      applyActiveFileTextState({ kind: result.kind === "markdown" ? "markdown" : "text", text: result.text, editable: Boolean(result.editable), version: typeof result.version === "string" ? result.version : "" });
      const renderPreview = viewMode === "preview" && currentActiveFileKind() === "markdown";
      const size = typeof result.size === "number" ? result.size : result.text.length;
      const statusParts = [path];
      if (renderPreview) statusParts.push("preview");
      if (!currentActiveFileEditable()) statusParts.push("read-only");
      statusParts.push(fmtBytes(size));
      return Object.freeze({ kind: "text", text: result.text, renderPreview, status: statusParts.join(" - ") });
    }

    function beginCompatibleVideoPreview(expectedToken = "") {
      const state = activeVideoFallback;
      const token = String(expectedToken || "");
      if (!state || (token && state.token !== token) || state.used || state.preparing) return null;
      state.preparing = true;
      applyFileMode();
      return activeVideoFallbackSnapshot();
    }

    function completeCompatibleVideoPreview(preview) {
      const token = preview && preview.token ? String(preview.token) : "";
      const state = activeVideoFallback;
      if (!state || (token && state.token !== token)) return false;
      state.used = true;
      state.preparing = false;
      applyFileMode();
      return true;
    }

    function failCompatibleVideoPreview(preview) {
      const token = preview && preview.token ? String(preview.token) : "";
      const state = activeVideoFallback;
      if (!state || (token && state.token !== token)) return false;
      state.preparing = false;
      applyFileMode();
      return true;
    }

    async function loadCompatibleVideoPreview(expectedToken = "", options = {}) {
      const preparePreview = requireFunction(options.preparePreview, "prepareCompatibleVideoPreview");
      const loadPreviewDom = requireFunction(options.loadPreviewDom, "loadCompatibleVideoPreviewDom");
      const errorText = requireFunction(options.errorText, "fileVideoPreviewErrorText");
      const explicit = Boolean(options.explicit);
      const state = beginCompatibleVideoPreview(expectedToken);
      if (!state) return false;
      const rel = state.rel || currentActiveFileIdentity().path || "video";
      fileStatus.textContent = explicit ? `${rel} - building compatible video preview...` : `${rel} - trying compatible video preview...`;
      try {
        await preparePreview(state.previewUrl);
        if (!completeCompatibleVideoPreview(state)) return false;
        fileStatus.textContent = `${rel} - loading compatible video preview...`;
        loadPreviewDom(state.previewUrl);
        return true;
      } catch (err) {
        if (failCompatibleVideoPreview(state)) {
          fileStatus.textContent = `${rel} - ${errorText(err)}`;
        }
        return false;
      }
    }

    function clearUsedCompatibleVideoPreview(token) {
      const state = activeVideoFallback;
      if (!state || state.token !== String(token || "") || !state.used) return false;
      activeVideoFallback = null;
      applyFileMode();
      return true;
    }

    function currentFileModeControlState() {
      const identity = currentActiveFileIdentity();
      const hasPath = Boolean(identity.path);
      const draft = Boolean(currentActiveFileDraft());
      const viewMode = currentFileViewMode();
      const entry = hasPath ? activeFileEntry() : null;
      const canToggleMode = Boolean(hasPath && !draft);
      const isDiff = viewMode === "diff";
      const isPreview = viewMode === "preview";
      const diffable = Boolean(canToggleMode && identity.gitPath && currentFileCandidateGitStateFresh() && entry && entry.changed && isDiffableFileKind(currentActiveFileKind()));
      const previewable = Boolean(!draft && currentActiveFileKind() === "markdown");
      const fallback = activeVideoFallback;
      const videoVisible = Boolean(fallback && fallback.previewUrl && !fallback.used);
      const videoPreparing = Boolean(fallback && fallback.preparing);
      const videoTitle = videoPreparing ? "Building compatible MP4 preview" : "Use compatible MP4 preview";
      return Object.freeze({
        diffActive: Boolean(hasPath && isDiff),
        previewActive: Boolean(hasPath && isPreview),
        diffDisabled: !diffable,
        previewDisabled: !canToggleMode,
        downloadDisabled: Boolean(!hasPath || draft),
        videoPreviewVisible: videoVisible,
        videoPreviewDisabled: Boolean(!videoVisible || videoPreparing),
        videoPreviewTitle: videoTitle,
        markdownPreviewVisible: previewable,
        shouldHidePasteDialog: viewMode !== "file",
        shouldExitEditMode: Boolean(viewMode !== "file" && currentFileEditMode()),
      });
    }

    function syncFileEditorReadOnly() {
      if (currentFileEditorKind() !== "file") return;
      const editor = focusEditor();
      if (!editor || typeof editor.updateOptions !== "function") return;
      editor.updateOptions({ readOnly: !activeFileEditorWritable() });
    }

    function updateFileEditButton() {
      const unavailable = isUnavailable();
      const canEdit = activeFileCanEnterEditMode();
      fileEditButton.disabled = unavailable || !canEdit;
      const savePending = isFileSavePending();
      const editMode = Boolean(currentFileEditMode());
      const dirty = Boolean(currentFileDirty());
      const saveStyle = editMode || savePending;
      fileEditButton.classList.toggle("active", saveStyle);
      fileEditButton.classList.toggle("primary", saveStyle);
      fileEditButton.classList.toggle("dirty", dirty);
      if (savePending) fileEditButton.innerHTML = iconSvg("save");
      else if (editMode) fileEditButton.innerHTML = iconSvg("save");
      else fileEditButton.innerHTML = iconSvg("edit");
      fileEditButton.title = unavailable ? "Session unavailable; copy edits before closing" : savePending ? "Saving file" : editMode ? "Save file" : canEdit ? "Edit file" : "File is read-only";
      fileEditButton.setAttribute("aria-label", unavailable ? "Session unavailable; copy edits before closing" : savePending ? "Saving file" : editMode ? "Save file" : "Edit file");
      updateFileTouchToolbar();
    }

    function isFileSavePending() {
      return Boolean(fileSavePending);
    }

    function currentFileDirty() {
      return fileDirty;
    }

    function setFileDirty(nextDirty) {
      fileDirty = Boolean(nextDirty);
      updateFileEditButton();
      updateFileTouchToolbar();
    }

    function clearActiveFileSaveState() {
      activeFileSaveToken = 0;
      fileSavePending = false;
    }

    function beginActiveFileSaveRequest() {
      const sessionId = currentSessionId();
      const identity = currentActiveFileIdentity();
      const path = identity.path;
      const apiPath = identity.apiPath || "";
      const draft = Boolean(currentActiveFileDraft());
      const gitPath = Boolean(identity.gitPath);
      const version = currentActiveFileVersion();
      const text = getFileEditorText();
      const token = ++fileSaveSeq;
      activeFileSaveToken = token;
      return Object.freeze({ sessionId, path, apiPath, draft, gitPath, version, text, token });
    }

    function isCurrentActiveFileSaveRequest(save) {
      const identity = currentActiveFileIdentity();
      return Boolean(
        save &&
          currentSessionId() === save.sessionId &&
          identity.path === save.path &&
          identity.apiPath === save.apiPath &&
          identity.gitPath === save.gitPath &&
          activeFileSaveToken === save.token &&
          !isUnavailable()
      );
    }

    function markActiveFileSavePending(save) {
      fileSavePending = true;
      updateFileEditButton();
      syncFileEditorReadOnly();
      fileStatus.textContent = `Saving ${save.path}...`;
    }

    function finishActiveFileSaveRequest(save) {
      if (!save || activeFileSaveToken !== save.token) return;
      clearActiveFileSaveState();
      syncFileEditorReadOnly();
      updateFileEditButton();
    }

    function buildActiveFileSaveBody(save) {
      const body = save.draft
        ? { path: save.path, text: save.text, create: true }
        : { path: save.path, text: save.text, version: save.version, git_path: save.gitPath };
      if (!save.draft && save.gitPath && save.apiPath) body.path_token = save.apiPath;
      return body;
    }

    function renderActiveFileSaveError(save, error) {
      if (error && error.status === 409) {
        renderSaveConflict(save.sessionId, save.path, error && error.message ? error.message : "conflict");
      } else {
        fileStatus.textContent = `save error: ${error && error.message ? error.message : "unknown error"}`;
      }
    }

    function applyActiveFileSaveSuccess(save, res, { exitEditMode = true } = {}) {
      const nextKind = String(currentActiveFileKind() || "text");
      const nextVersion = res && typeof res.version === "string" ? res.version : currentActiveFileVersion();
      const nextEditable = res && typeof res.editable === "boolean" ? res.editable : currentActiveFileEditable();
      applyActiveFileTextState({ kind: nextKind, text: save.text, editable: nextEditable, version: nextVersion, draft: false });
      if (save.draft) {
        setActiveFileIdentity(save.path, { line: currentActiveFileLine(), gitPath: false, apiPath: "" });
      }
      applyFileMode();
      setFileDirty(false);
      if (exitEditMode) setFileEditMode(false);
      const size = res && typeof res.size === "number" ? res.size : save.text.length;
      fileStatus.textContent = `${save.path} - ${fmtBytes(size)}`;
      rememberOpenedFile(save.path, res && typeof res.path === "string" ? res.path : null);
      renderFilePickerMenu();
      return true;
    }

    async function submitActiveFileSave(save, { exitEditMode = true } = {}) {
      const saveStillCurrent = () => isCurrentActiveFileSaveRequest(save);
      markActiveFileSavePending(save);
      try {
        const saveBody = buildActiveFileSaveBody(save);
        const res = await api(`/api/sessions/${save.sessionId}/file/write`, {
          method: "POST",
          body: saveBody,
        });
        if (!saveStillCurrent()) return true;
        return applyActiveFileSaveSuccess(save, res, { exitEditMode });
      } catch (error) {
        if (!saveStillCurrent()) return false;
        renderActiveFileSaveError(save, error);
        return false;
      } finally {
        finishActiveFileSaveRequest(save);
      }
    }

    async function saveActiveFileEdits({ exitEditMode = true } = {}) {
      if (blockUnavailableFileAction()) return false;
      const identity = currentActiveFileIdentity();
      if (!currentSessionId() || !identity.path || !isTextFileKind(currentActiveFileKind()) || !currentActiveFileEditable()) return false;
      if (!currentFileDirty() && !currentActiveFileDraft()) {
        if (exitEditMode) setFileEditMode(false);
        return true;
      }
      const save = beginActiveFileSaveRequest();
      return await submitActiveFileSave(save, { exitEditMode });
    }

    function prepareFileEditorTextRestore(text) {
      const restoredText = String(text || "");
      if (currentFileEditorKind() !== "file") {
        setFileDirty(false);
        return Object.freeze({ kind: "skip" });
      }
      return Object.freeze({ kind: "restore", text: restoredText });
    }

    function finishFileEditorTextRestore() {
      setFileDirty(false);
    }

    function discardActiveFileEdits() {
      restoreFileEditorText(currentActiveFileText());
      setFileEditMode(false);
    }

    function isFileUnsavedPromptPending() {
      return Boolean(fileUnsavedPromptResolver);
    }

    function fileUnsavedPromptPlan() {
      if (!currentFileDirty()) return Object.freeze({ kind: "choice", choice: "discard" });
      if (fileUnsavedPromptResolver) return Object.freeze({ kind: "choice", choice: "cancel" });
      return Object.freeze({ kind: "prompt" });
    }

    function beginFileUnsavedPrompt() {
      const plan = fileUnsavedPromptPlan();
      if (plan.kind === "choice") return Promise.resolve(plan.choice);
      return new Promise((resolve) => {
        fileUnsavedPromptResolver = resolve;
      });
    }

    function resolveFileUnsavedPrompt(choice = "cancel") {
      const resolve = fileUnsavedPromptResolver;
      fileUnsavedPromptResolver = null;
      if (!resolve) return false;
      resolve(String(choice || "cancel"));
      return true;
    }

    function applyPlainTextFallbackState() {
      setFileEditMode(false);
      setFileDirty(false);
      updateFileEditButton();
      updateFileTouchToolbar();
    }

    async function maybeHandleUnsavedFileChanges() {
      if (!currentFileDirty()) return true;
      const choice = await promptUnsavedFileChoice();
      if (choice === "discard") {
        discardActiveFileEdits();
        return true;
      }
      if (choice === "save") return await saveActiveFileEdits({ exitEditMode: true });
      return false;
    }

    function handleFileUnsavedSaveChoice() {
      if (blockUnavailableFileAction()) return false;
      hideFileUnsavedDialog("save");
      return true;
    }

    function handleFileUnsavedDiscardChoice() {
      hideFileUnsavedDialog("discard");
      return true;
    }

    function handleFileUnsavedCancelChoice() {
      hideFileUnsavedDialog("cancel");
      return true;
    }

    async function setFileViewModeWithGuard(mode) {
      if (blockUnavailableFileAction()) return false;
      const next = mode === "preview" ? "preview" : mode === "file" ? "file" : "diff";
      if (next === currentFileViewMode()) return true;
      if (currentActiveFileDraft() && next !== "file") return false;
      if (!(await maybeHandleUnsavedFileChanges())) return false;
      if (blockUnavailableFileAction()) return false;
      setFileViewMode(next);
      renderFilePickerMenu();
      const identity = currentActiveFileIdentity();
      await openFilePath(identity.path, { line: activeFileLine, gitPath: identity.gitPath, apiPath: identity.apiPath });
      return true;
    }

    async function requestHideFileViewer() {
      if (!(await maybeHandleUnsavedFileChanges())) return false;
      hideFileViewer();
      return true;
    }

    async function openFilePathWithGuard(path, { line = null, mode = null, isCurrent = null, gitPath = false, apiPath = "" } = {}) {
      if (blockUnavailableFileAction()) return false;
      const sessionAtStart = currentFileSessionId();
      const currentGuard = typeof isCurrent === "function" ? isCurrent : () => currentFileSessionId() === sessionAtStart && !isFileViewerSessionUnavailable();
      if (!(await maybeHandleUnsavedFileChanges())) return false;
      if (blockUnavailableFileAction()) return false;
      if (!currentGuard()) return false;
      const openMode = normalizeExplicitFileOpenMode(mode);
      setFilePath(path, { line, gitPath, apiPath });
      if (openMode) setFileViewMode(openMode);
      renderFilePickerMenu();
      await openFilePath(path, { line, gitPath, apiPath, mode: openMode });
      return Boolean(currentGuard());
    }

    async function openFilePathWithResolvedMode(path, { line = null, changed = null, isCurrent = null, gitPath = null, apiPath = "" } = {}) {
      if (blockUnavailableFileAction()) return false;
      const sessionAtStart = currentFileSessionId();
      const currentGuard = typeof isCurrent === "function" ? isCurrent : () => currentFileSessionId() === sessionAtStart && !isFileViewerSessionUnavailable();
      const token = normalizeFileApiPath(apiPath);
      const useGitPath = gitPath === null || gitPath === undefined ? isGitFileCandidatePath(path, changed, null, token) : Boolean(gitPath);
      const entry = fileEntryForPath(path, useGitPath, token);
      const requestApiPath = token || normalizeFileApiPath(entry && entry.apiPath);
      let mode;
      try {
        mode = await resolveFileOpenMode(path, { changed, gitPath: useGitPath, apiPath: requestApiPath });
      } catch (error) {
        if (blockUnavailableFileAction()) return false;
        throw error;
      }
      if (!currentGuard()) return false;
      return await openFilePathWithGuard(path, { line, mode, isCurrent: currentGuard, gitPath: useGitPath, apiPath: requestApiPath });
    }

    async function openDraftFilePathWithGuard(path) {
      if (blockUnavailableFileAction()) return false;
      const rel = normalizeDraftFilePath(path);
      if (!rel) {
        fileStatus.textContent = "Choose a valid relative file path.";
        return false;
      }
      if (!(await maybeHandleUnsavedFileChanges())) return false;
      if (blockUnavailableFileAction()) return false;
      try {
        const inspect = await inspectSessionFilePath(rel);
        if (blockUnavailableFileAction()) return false;
        if (inspect && inspect.exists) {
          if (inspect.kind === "directory") {
            fileStatus.textContent = `${rel} - path is a directory`;
            return false;
          }
          return await openFilePathWithGuard(rel, { line: null, mode: "file" });
        }
      } catch (error) {
        if (blockUnavailableFileAction()) return false;
        fileStatus.textContent = `error: ${error && error.message ? error.message : "unable to inspect path"}`;
        return false;
      }
      if (blockUnavailableFileAction()) return false;
      setFileViewMode("file");
      setFilePath(rel, { line: null, gitPath: false });
      renderFilePickerMenu();
      await openDraftFilePath(rel, { line: null });
      return true;
    }

    async function openDraftFilePath(path, { line = null } = {}) {
      if (blockUnavailableFileAction()) return;
      if (!normalizeSessionId(currentSessionId())) return;
      const openRequest = startFileOpenRequest(path, { line, gitPath: false });
      const request = openRequest.request;
      const rel = normalizeDraftFilePath(path);
      if (!rel) {
        fileStatus.textContent = "Choose a valid relative file path.";
        openRequest.done();
        return;
      }
      fileStatus.textContent = "Preparing new file...";
      resetFileViewerPanel();
      try {
        const loaded = await applyDraftFileLoad(rel, request);
        if (!loaded) return;
      } catch (error) {
        renderDraftFileOpenError(request, error);
        return;
      } finally {
        openRequest.done();
      }
    }

    function finalizeFileOpenSuccess(rel, absPath = null) {
      applyFileMode();
      rememberOpenedFile(rel, absPath);
      rememberActiveFileSelection();
      updateFileEditButton();
      renderFilePickerMenu();
      return true;
    }

    function clearFileTouchSelectionState() {
      fileTouchSelectMode = false;
      fileTouchSelectAnchor = null;
      fileTouchSelectHead = null;
      fileTouchSelectGoalColumn = null;
    }

    function currentFileTouchSelectMode() {
      return fileTouchSelectMode;
    }

    function isFileTouchToolbarActive() {
      return Boolean(
        useTouchFileEditorControls() &&
          isFileViewerOpen() &&
          isTextFileKind(currentActiveFileKind()) &&
          currentFileViewMode() !== "preview" &&
          hasActiveFileCodeEditor()
      );
    }

    function currentFileTouchToolbarState() {
      const visible = isFileTouchToolbarActive();
      const selectActive = Boolean(currentFileTouchSelectMode());
      if (!visible) return Object.freeze({ visible: false, selectActive, dpadVisible: false, copyVisible: false, pasteVisible: false });
      return Object.freeze({
        visible: true,
        selectActive,
        dpadVisible: selectActive,
        copyVisible: Boolean(getActiveFileSelectionText()),
        pasteVisible: activeFileEditorIdleTextWritable(),
      });
    }

    function fileDiffSelectionHideOptions() {
      return fileTouchSelectMode
        ? { enabled: false }
        : {
            enabled: true,
            contextLineCount: 4,
            minimumLineCount: 1,
            revealLineCount: 2,
          };
    }

    function syncFileDiffSelectionMode() {
      updateFileDiffEditorOptions({ hideUnchangedRegions: fileDiffSelectionHideOptions() });
    }

    function resetFileTouchSelectionState({ collapse = false } = {}) {
      const editor = collapse ? focusEditor() : null;
      const cursor = editor ? normalizeFileEditorPosition(editor, editor.getPosition && editor.getPosition()) : null;
      clearFileTouchSelectionState();
      if (editor && cursor) applyFileEditorSelection(editor, cursor, null);
      syncFileEditorReadOnly();
      syncFileDiffSelectionMode();
      updateFileTouchToolbar();
    }

    function toggleFileTouchSelectionMode() {
      if (fileTouchSelectMode) {
        resetFileTouchSelectionState({ collapse: true });
        focusActiveFileCodeEditor();
        return;
      }
      const editor = focusEditor();
      if (!editor) return;
      const cursor = normalizeFileEditorPosition(editor, editor.getPosition && editor.getPosition()) || { lineNumber: 1, column: 1 };
      fileTouchSelectMode = true;
      fileTouchSelectAnchor = { ...cursor };
      fileTouchSelectHead = { ...cursor };
      fileTouchSelectGoalColumn = cursor.column;
      applyFileEditorSelection(editor, cursor, cursor);
      syncFileEditorReadOnly();
      syncFileDiffSelectionMode();
      updateFileTouchToolbar();
      focusActiveFileCodeEditor();
    }

    function handleFileTouchMoveButtonPress(direction) {
      focusActiveFileCodeEditor();
      moveFileTouchSelection(direction);
    }

    function moveFileTouchSelection(direction) {
      if (!fileTouchSelectMode) return;
      const editor = focusEditor();
      if (!editor || typeof editor.trigger !== "function") {
        setToast("selection move unavailable");
        return;
      }
      const args =
        direction === "left"
          ? { to: "left", by: "character", value: 1, select: true }
          : direction === "right"
            ? { to: "right", by: "character", value: 1, select: true }
            : direction === "up"
              ? { to: "up", by: "wrappedLine", value: 1, select: true }
              : direction === "down"
                ? { to: "down", by: "wrappedLine", value: 1, select: true }
                : null;
      if (!args) return;
      try {
        editor.trigger("file-touch-select", "cursorMove", args);
        const pos = normalizeFileEditorPosition(editor, editor.getPosition && editor.getPosition());
        if (pos) {
          fileTouchSelectHead = { ...pos };
          fileTouchSelectGoalColumn = pos.column;
        }
        focusActiveFileCodeEditor();
        updateFileTouchToolbar();
      } catch (error) {
        setToast(`selection move error: ${error && error.message ? error.message : "unknown error"}`);
      }
    }

    function fileEditorShortcutBlocked(target) {
      if (!isFileViewerOpen()) return true;
      if (hasBlockingFileEditorModal()) return true;
      if (target && isTextEntryTarget(target) && !isActiveFileEditorInput(target)) return true;
      return false;
    }

    function handleFileTouchSelectionKeydown(event) {
      const e = event || {};
      if (!currentFileTouchSelectMode() || !isFileTouchToolbarActive()) return;
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
      const target = eventTargetElement(e.target);
      if (fileEditorShortcutBlocked(target)) return;
      if (target && !target.closest("#fileViewer")) return;
      const key = String(e.key || "").toLowerCase();
      if (key === "escape") {
        e.preventDefault();
        e.stopPropagation();
        resetFileTouchSelectionState({ collapse: true });
        return;
      }
      const direction = key === "h" ? "left" : key === "j" ? "down" : key === "k" ? "up" : key === "l" ? "right" : "";
      if (!direction) {
        const blocksEdit =
          key === "enter" ||
          key === "tab" ||
          key === " " ||
          key === "backspace" ||
          key === "delete" ||
          (key.length === 1 && !e.altKey && !e.ctrlKey && !e.metaKey);
        if (!blocksEdit) return;
        e.preventDefault();
        e.stopPropagation();
        return;
      }
      e.preventDefault();
      e.stopPropagation();
      moveFileTouchSelection(direction);
    }

    function handleFileEditorDeleteKeydown(event) {
      const e = event || {};
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return false;
      const key = String(e.key || "").toLowerCase();
      const command = fileEditorDeleteCommandForKey(key);
      if (!command) return false;
      if (!activeFileEditorWritable()) return false;
      const target = eventTargetElement(e.target);
      if (fileEditorShortcutBlocked(target)) return false;
      if (!isActiveFileEditorInput(target)) return false;
      const editor = focusEditor();
      if (!editor || typeof editor.trigger !== "function") return false;
      fileTouchDeleteNativeSuppressUntil = nowMs() + 250;
      e.preventDefault();
      e.stopPropagation();
      try {
        focusActiveFileCodeEditor();
        editor.trigger("file-editor-delete-key", command, null);
        if (currentFileTouchSelectMode()) resetFileTouchSelectionState();
        return true;
      } catch (error) {
        setToast(`delete error: ${error && error.message ? error.message : "unknown error"}`);
        return true;
      }
    }

    function isFileEditorNativeDeleteEvent(event) {
      const inputType = String((event && event.inputType) || "");
      if (inputType !== "deleteContentBackward" && inputType !== "deleteContentForward") return false;
      return isActiveFileEditorInput(eventTargetElement(event && event.target));
    }

    function suppressFileEditorNativeDelete(event) {
      if (nowMs() > fileTouchDeleteNativeSuppressUntil || !isFileEditorNativeDeleteEvent(event)) return false;
      if (event.cancelable) event.preventDefault();
      event.stopPropagation();
      fileTouchDeleteNativeSuppressUntil = 0;
      return true;
    }

    function insertIntoActiveFileEditor(text) {
      if (!activeFileEditorIdleWritable()) return false;
      const editor = focusEditor();
      if (!editor || !fileEditorEditSupportAvailable() || typeof editor.executeEdits !== "function") return false;
      const current = normalizeFileEditorPosition(editor, editor.getPosition && editor.getPosition()) || { lineNumber: 1, column: 1 };
      const selection = editor.getSelection && editor.getSelection();
      const range = selection && !isCollapsedFileSelection(selection)
        ? {
            startLineNumber: selection.startLineNumber,
            startColumn: selection.startColumn,
            endLineNumber: selection.endLineNumber,
            endColumn: selection.endColumn,
          }
        : {
            startLineNumber: current.lineNumber,
            startColumn: current.column,
            endLineNumber: current.lineNumber,
            endColumn: current.column,
          };
      if (typeof editor.pushUndoStop === "function") editor.pushUndoStop();
      editor.executeEdits("file-touch-paste", [{ range, text: String(text || ""), forceMoveMarkers: true }]);
      const nextCursor = positionAfterInsertedText({ lineNumber: range.startLineNumber, column: range.startColumn }, text);
      resetFileTouchSelectionState();
      applyFileEditorSelection(editor, nextCursor, null);
      if (typeof editor.pushUndoStop === "function") editor.pushUndoStop();
      setFileDirty(getFileEditorText() !== String(currentActiveFileText() || ""));
      focusActiveFileCodeEditor();
      return true;
    }

    function requestManualFilePasteDialog() {
      if (!activeFileEditorIdleTextWritable()) return false;
      return showFilePasteDialog();
    }

    async function pasteFromClipboardIntoActiveFile() {
      if (!activeFileEditorIdleTextWritable()) return false;
      if (!clipboardReadAvailable()) {
        if (requestManualFilePasteDialog()) setToast("paste manually");
        else {
          setToast("paste unavailable");
          focusActiveFileCodeEditor();
        }
        return false;
      }
      try {
        const text = await readClipboardText();
        if (blockUnavailableFileAction()) return false;
        if (!text) {
          setToast("clipboard empty");
          focusActiveFileCodeEditor();
          return false;
        }
        if (!insertIntoActiveFileEditor(text)) {
          setToast("paste unavailable");
          focusActiveFileCodeEditor();
          return false;
        }
        setToast("pasted");
        focusActiveFileCodeEditor();
        return true;
      } catch (error) {
        if (requestManualFilePasteDialog()) setToast("paste manually");
        else {
          setToast(`paste error: ${error && error.message ? error.message : "clipboard denied"}`);
          focusActiveFileCodeEditor();
        }
        return false;
      }
    }

    function handleFilePasteInsert(text) {
      if (blockUnavailableFileAction()) return false;
      if (!insertIntoActiveFileEditor(text)) return false;
      hideFilePasteDialog();
      setToast("text inserted");
      return true;
    }

    async function copyActiveFileSelection() {
      const text = getActiveFileSelectionText();
      if (!text) {
        setToast("nothing selected");
        return false;
      }
      try {
        await copyToClipboard(text);
        resetFileTouchSelectionState({ collapse: true });
        setToast("selection copied");
        focusActiveFileCodeEditor();
        return true;
      } catch (error) {
        setToast(`copy error: ${error && error.message ? error.message : "unknown error"}`);
        focusActiveFileCodeEditor();
        return false;
      }
    }

    async function handleFileDiffModeButtonPress() {
      const nextMode = currentFileViewMode() === "diff" ? currentFileNonDiffMode() : "diff";
      return await setFileViewModeWithGuard(nextMode);
    }

    async function handleFilePreviewModeButtonPress() {
      const identity = currentActiveFileIdentity();
      if (!isMarkdownPreviewable(identity.path)) return false;
      const nextMode = currentFileViewMode() === "preview" ? "file" : "preview";
      return await setFileViewModeWithGuard(nextMode);
    }

    async function handleFileEditButtonPress() {
      if (isFileSavePending()) return false;
      if (currentFileEditMode()) {
        await saveActiveFileEdits({ exitEditMode: true });
        return true;
      }
      if (currentFileViewMode() !== "file") {
        const changed = await setFileViewModeWithGuard("file");
        if (!changed) return false;
      }
      if (!currentActiveFileEditable() || !isTextFileKind(currentActiveFileKind())) return false;
      setFileEditMode(true);
      return true;
    }

    function handleFileEditorSaveShortcut(event) {
      if (!event || event.defaultPrevented || event.isComposing) return false;
      const key = String(event.key || "").toLowerCase();
      if (key !== "s" || !(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) return false;
      const target = eventTargetElement(event.target);
      if (fileEditorShortcutBlocked(target)) return false;
      if (!activeFileEditorIdleTextWritable()) return false;
      const sessionId = normalizeSessionId(currentSessionId());
      const identity = currentActiveFileIdentity();
      if (!sessionId || !identity.path) return false;
      event.preventDefault();
      event.stopPropagation();
      void saveActiveFileEdits({ exitEditMode: false });
      return true;
    }

    async function handleFileVideoPreviewButtonPress(token, loadPreview) {
      const loadCompatiblePreview = requireFunction(loadPreview, "loadCompatibleVideoPreview");
      return await loadCompatiblePreview(token || "", { explicit: true });
    }

    function activeFileDownloadApiPath() {
      if (blockUnavailableFileAction()) return "";
      const sessionId = normalizeSessionId(currentSessionId());
      const identity = currentActiveFileIdentity();
      if (!sessionId || !identity.path) return "";
      const tokenQuery = identity.gitPath && identity.apiPath ? `&path_token=${encodeURIComponent(identity.apiPath)}` : "";
      return `/api/sessions/${sessionId}/file/download?path=${encodeURIComponent(identity.path)}${tokenQuery}${identity.gitPath ? "&git_path=1" : ""}`;
    }

    async function openFilePath(nextPath = null, { line = undefined, gitPath = undefined, apiPath = undefined, mode = null } = {}) {
      if (blockUnavailableFileAction()) return false;
      if (!normalizeSessionId(currentSessionId())) return false;
      const openRequest = startFileOpenRequest(nextPath, { line, gitPath, apiPath });
      const request = openRequest.request;
      const rel = openRequest.path;
      if (!rel) {
        fileStatus.textContent = "Choose a file first.";
        openRequest.done();
        return false;
      }
      fileStatus.textContent = "Loading...";
      resetFileViewerPanel();
      try {
        const viewMode = resolveFileOpenViewMode(request, rel, mode);
        if (viewMode !== currentFileViewMode()) setFileViewMode(viewMode);
        const openResult = await fetchFileOpenResult(request, rel, viewMode);
        if (!isCurrentFileOpenRequest(request)) return false;
        const loaded = await applyFileLoadResult(rel, openResult.result, request, { viewMode });
        if (!loaded) return false;
        return finalizeFileOpenSuccess(rel, openResult.absPath);
      } catch (error) {
        return renderFileOpenError(request, error);
      } finally {
        openRequest.done();
      }
    }

    async function applyDraftFileLoad(rel, request) {
      if (currentFileViewMode() !== "file") setFileViewMode("file");
      applyActiveFileTextState({ text: "", editable: true, version: "", draft: true });
      applyFileMode();
      const rendered = await renderMonacoFile(rel, "", request.line, "", request);
      if (!rendered || !isCurrentFileOpenRequest(request)) return false;
      setFileEditMode(true);
      fileStatus.textContent = `${rel} - new file`;
      rememberActiveFileSelection();
      renderFilePickerMenu();
      return true;
    }

    function renderFileOpenError(request, error) {
      if (isFileOpenAbortError(error)) return false;
      if (!isCurrentFileOpenRequest(request)) return false;
      resetActiveFileBufferState();
      fileStatus.textContent = `error: ${error && error.message ? error.message : "unknown error"}`;
      updateFileTouchToolbar();
      return false;
    }

    function renderDraftFileOpenError(request, error) {
      if (isFileOpenAbortError(error)) return false;
      if (!isCurrentFileOpenRequest(request)) return false;
      resetActiveFileBufferState();
      fileStatus.textContent = `error: ${error && error.message ? error.message : "unknown error"}`;
      return false;
    }

    async function fetchFileOpenResult(request, rel, viewMode) {
      if (viewMode === "diff") {
        const pathTokenQuery = request.apiPath ? `&path_token=${encodeURIComponent(request.apiPath)}` : "";
        const res = await api(`/api/sessions/${request.sessionId}/git/file_versions?path=${encodeURIComponent(rel)}${pathTokenQuery}`, {
          signal: request.signal,
        });
        return Object.freeze({
          result: Object.freeze({
            kind: "diff",
            baseText: res && typeof res.base_text === "string" ? res.base_text : "",
            currentText: res && typeof res.current_text === "string" ? res.current_text : "",
            baseExists: res && res.base_exists,
            currentExists: res && res.current_exists,
          }),
          absPath: res && typeof res.abs_path === "string" ? res.abs_path : null,
        });
      }
      const gitPathQuery = request.gitPath ? "&git_path=1" : "";
      const pathTokenQuery = request.gitPath && request.apiPath ? `&path_token=${encodeURIComponent(request.apiPath)}` : "";
      const res = await api(`/api/sessions/${request.sessionId}/file/read?path=${encodeURIComponent(rel)}${pathTokenQuery}${gitPathQuery}`, {
        signal: request.signal,
      });
      return Object.freeze({
        result: res,
        absPath: res && typeof res.path === "string" ? res.path : null,
      });
    }

    function isSaveConflictCurrent(conflict) {
      return Boolean(conflict && currentSessionId() === conflict.sessionId && activeFilePath === conflict.path && !isUnavailable());
    }

    async function reloadSaveConflict(conflict) {
      if (!isSaveConflictCurrent(conflict)) return;
      const savePath = conflict.path;
      const ok = confirmReload(`Reload ${savePath} from disk and discard your unsaved editor draft?`);
      if (!ok) return;
      fileStatus.textContent = `Reloading ${savePath}...`;
      const reloaded = await openFilePath(savePath, { line: activeFileLine, gitPath: activeFileGitPath, apiPath: activeFileApiPath });
      if (!reloaded && isSaveConflictCurrent(conflict)) fileStatus.textContent = `${savePath} - reload failed`;
    }

    function keepEditingSaveConflict(conflict) {
      if (!isSaveConflictCurrent(conflict)) return;
      const savePath = conflict.path;
      fileStatus.textContent = `${savePath} - editing unsaved conflict`;
      const editor = focusEditor();
      if (editor && typeof editor.focus === "function") editor.focus();
    }

    function handleSaveConflictActionEvent(event, action) {
      event.preventDefault();
      event.stopPropagation();
      return action();
    }

    function renderSaveConflict(saveSessionId, savePath, message = "conflict") {
      const conflict = fileSaveConflictTarget(saveSessionId, savePath);
      activeSaveConflict = conflict;
      const label = el("span", { class: "fileConflictText", text: `${savePath} - save conflict: ${message}` });
      const reloadBtn = el("button", {
        class: "icon-btn text-btn fileConflictReload",
        type: "button",
        text: "Reload from disk",
        title: "Discard unsaved edits and load the current disk version",
      });
      const keepBtn = el("button", {
        class: "icon-btn text-btn fileConflictKeep",
        type: "button",
        text: "Keep editing",
        title: "Keep the unsaved draft in the editor",
      });
      reloadBtn.onclick = (event) => handleSaveConflictActionEvent(event, () => reloadSaveConflict(conflict));
      keepBtn.onclick = (event) => handleSaveConflictActionEvent(event, () => keepEditingSaveConflict(conflict));
      const actions = el("span", { class: "fileConflictActions" }, [reloadBtn, keepBtn]);
      fileStatus.replaceChildren(label, actions);
      return conflict;
    }

    function currentSaveConflict() {
      return activeSaveConflict;
    }

    return Object.freeze({
      renderSaveConflict,
      reloadSaveConflict,
      keepEditingSaveConflict,
      isSaveConflictCurrent,
      currentSaveConflict,
      isFileViewerSessionUnavailable,
      clearFileViewerUnavailableSession,
      disableFileViewerForUnavailableSession,
      handleFileViewerSessionUnavailable,
      isFileSavePending,
      currentFileDirty,
      setFileDirty,
      clearActiveFileSaveState,
      beginActiveFileSaveRequest,
      isCurrentActiveFileSaveRequest,
      markActiveFileSavePending,
      finishActiveFileSaveRequest,
      buildActiveFileSaveBody,
      renderActiveFileSaveError,
      applyActiveFileSaveSuccess,
      submitActiveFileSave,
      saveActiveFileEdits,
      discardActiveFileEdits,
      isFileUnsavedPromptPending,
      fileUnsavedPromptPlan,
      beginFileUnsavedPrompt,
      resolveFileUnsavedPrompt,
      applyPlainTextFallbackState,
      maybeHandleUnsavedFileChanges,
      handleFileUnsavedSaveChoice,
      handleFileUnsavedDiscardChoice,
      handleFileUnsavedCancelChoice,
      setFileViewModeWithGuard,
      requestHideFileViewer,
      openFilePathWithGuard,
      openFilePath,
      openDraftFilePathWithGuard,
      openDraftFilePath,
      nextActiveFileIdentity,
      currentActiveFileIdentity,
      currentActiveFileLine,
      rememberActiveFileSelection,
      preferredFileSelectionForSession,
      fileCandidateKey,
      fileCandidateKeyForEntry,
      cloneFileCandidateEntry,
      applyFileCandidateEntries,
      currentFileCandidateKeys,
      currentFileCandidateEntries,
      fileEntryForKey,
      fileEntryForPath,
      fileApiPathForPath,
      activeFileEntry,
      isGitFileCandidatePath,
      currentFileCandidateGitStateFresh,
      setFileCandidateGitStateFresh,
      rememberFileCandidateCache,
      fileCandidateCacheEntry,
      deleteFileCandidateCache,
      fileCandidateCacheSize,
      applyFileCandidateRefreshEntries,
      clearFileCandidateRefreshEntries,
      applyFreshFileCandidateCache,
      upsertFileEntry,
      pickerEntryForKey,
      pickerEntryForPath,
      resolveFileViewerOpenTarget,
      currentFileViewerSessionId,
      setFileViewerSessionId,
      clearFileViewerSessionId,
      beginFileViewerSessionSync,
      invalidateFileViewerSessionSync,
      isCurrentFileViewerSessionSync,
      beginFileCandidateRefresh,
      isCurrentFileCandidateRefresh,
      currentFileEditMode,
      currentFileEditorKind,
      isFileEditorProgrammaticChange,
      beginFileEditorProgrammaticChange,
      finishFileEditorProgrammaticChange,
      runFileEditorProgrammaticChange,
      setFileEditorKind,
      prepareFileEditorTextRestore,
      finishFileEditorTextRestore,
      setFileEditMode,
      currentActiveFileKind,
      currentActiveFileText,
      currentActiveFileEditable,
      currentActiveFileVersion,
      currentActiveFileDraft,
      resetActiveFileBufferState,
      applyActiveFileTextState,
      applyActiveFileDiffState,
      applyActiveFileNonTextState,
      clearActiveFileIdentity,
      setActiveFileIdentity,
      beginActiveFileIdentity,
      abortPendingFileOpenTransport,
      cancelPendingFileOpen,
      beginFileOpenRequest,
      isCurrentFileOpenRequest,
      finalizeFileOpenRequest,
      startFileOpenRequest,
      normalizeExplicitFileOpenMode,
      resolveFileOpenViewMode,
      resolveFileOpenMode,
      openFilePathWithResolvedMode,
      fetchFileOpenResult,
      isFileOpenAbortError,
      blockUnavailableFileAction,
      currentFileEditorState,
      fileEditorCapabilities,
      activeFileEditorCapabilities,
      activeFileCanEnterEditMode,
      activeFileEditorWritable,
      activeFileEditorIdleWritable,
      activeFileEditorIdleTextWritable,
      activeFileEditModeAllowedInCurrentView,
      currentFileViewMode,
      setFileViewerReturnFocusElement,
      takeFileViewerReturnFocusElement,
      setFileUnsavedReturnFocusElement,
      takeFileUnsavedReturnFocusElement,
      currentFileNonDiffMode,
      setFileViewMode,
      setActiveVideoFallback,
      clearActiveVideoFallback,
      currentActiveVideoFallback,
      setActivePdfRenderState,
      takeActivePdfRenderState,
      clearActivePdfRenderState,
      isActivePdfRenderState,
      currentActiveVideoPreviewToken,
      prepareActiveVideoLoadResult,
      handleActiveVideoLoadError,
      handleActiveVideoLoadedMetadata,
      prepareFileLoadResult,
      beginCompatibleVideoPreview,
      completeCompatibleVideoPreview,
      failCompatibleVideoPreview,
      loadCompatibleVideoPreview,
      clearUsedCompatibleVideoPreview,
      currentFileModeControlState,
      syncFileEditorReadOnly,
      updateFileEditButton,
      clearFileTouchSelectionState,
      currentFileTouchSelectMode,
      currentFileTouchToolbarState,
      resetFileTouchSelectionState,
      toggleFileTouchSelectionMode,
      handleFileTouchMoveButtonPress,
      moveFileTouchSelection,
      handleFileTouchSelectionKeydown,
      handleFileEditorDeleteKeydown,
      suppressFileEditorNativeDelete,
      insertIntoActiveFileEditor,
      pasteFromClipboardIntoActiveFile,
      handleFilePasteInsert,
      copyActiveFileSelection,
      handleFileDiffModeButtonPress,
      handleFilePreviewModeButtonPress,
      handleFileEditButtonPress,
      handleFileEditorSaveShortcut,
      handleFileVideoPreviewButtonPress,
      activeFileDownloadApiPath,
      finalizeFileOpenSuccess,
      applyDraftFileLoad,
      renderFileOpenError,
      renderDraftFileOpenError,
    });
  }

  window.CodoxearFileViewer = Object.freeze({
    bindFileTouchClick,
    bindFileTouchPress,
    createFilePasteDialogRuntime,
    createFileViewerController,
    createPdfLoader,
  });
})();
