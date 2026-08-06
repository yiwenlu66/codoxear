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
  const FILE_EDITOR_UNAVAILABLE_MESSAGE = "Editing is unavailable because the code editor failed to load. Read-only preview remains available.";
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
        apiPath: apiPath === undefined ? (useGitPath ? fileApiPathForPath(rel, reusableApiPath) : normalizeFileApiPath(reusableApiPath)) : normalizeFileApiPath(apiPath),
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
    const { createFileCandidateStateRuntime } = window.CodoxearFileCandidateState;
    const candidateState = createFileCandidateStateRuntime({
      normalizeFileApiPath,
      normalizeLineNumber,
      normalizeSessionId,
      preferredFileSelectionForSession,
      currentActiveFileIdentity,
      applyFileMode,
    });
    const {
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
      currentFileCandidateGitStateMessage,
      setFileCandidateGitStateMessage,
      clearFileCandidateGitStateMessage,
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
    } = candidateState;
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
      const editorSupportsWrite = editorKind !== "plain-fallback";
      const canEnterEditMode = Boolean(!unavailable && viewMode === "file" && String(state.path || "") && !savePending && (!kind || textKind) && editorSupportsWrite && editable);
      const writable = Boolean(editMode && editable && viewMode === "file" && !unavailable && editorSupportsWrite);
      const idleWritable = Boolean(writable && !savePending);
      const idleTextWritable = Boolean(idleWritable && textKind);
      const editModeAllowedInCurrentView = Boolean(viewMode === "file" && textKind && editable && !unavailable && editorSupportsWrite);
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
    const { createFileViewerOperationsRuntime } = window.CodoxearFileViewerOperations;
    const operations = createFileViewerOperationsRuntime({ ...deps,
      normalizeSessionId, isUnavailable, blockUnavailableFileAction,
      currentFileViewMode, currentFileNonDiffMode, setFileViewMode, currentFileEditMode, currentFileEditorKind, setFileEditorKind, setFileEditMode,
      currentActiveFileKind, currentActiveFileText, currentActiveFileEditable, currentActiveFileVersion, currentActiveFileDraft,
      applyActiveFileTextState, applyActiveFileDiffState, applyActiveFileNonTextState, currentActiveFileIdentity, currentActiveFileLine,
      startFileOpenRequest, isCurrentFileOpenRequest, normalizeExplicitFileOpenMode, resolveFileOpenMode, isFileOpenAbortError,
      activeFileEntry, isGitFileCandidatePath, currentFileCandidateGitStateFresh, activeFileCanEnterEditMode, activeFileEditorWritable, activeFileEditorIdleTextWritable, currentFileEditorState,
      fileEntryForPath,
    });
    const { setActiveVideoFallback, clearActiveVideoFallback, currentActiveVideoFallback, setActivePdfRenderState, takeActivePdfRenderState, clearActivePdfRenderState, isActivePdfRenderState, disposeActivePdfRender, currentActiveVideoPreviewToken, prepareActiveVideoLoadResult, handleActiveVideoLoadError, handleActiveVideoLoadedMetadata, prepareFileLoadResult, beginCompatibleVideoPreview, completeCompatibleVideoPreview, failCompatibleVideoPreview, loadCompatibleVideoPreview, clearUsedCompatibleVideoPreview, currentFileModeControlState, syncFileEditorReadOnly, updateFileEditButton, isFileSavePending, currentFileDirty, setFileDirty, clearActiveFileSaveState, beginActiveFileSaveRequest, isCurrentActiveFileSaveRequest, markActiveFileSavePending, finishActiveFileSaveRequest, buildActiveFileSaveBody, renderActiveFileSaveError, applyActiveFileSaveSuccess, submitActiveFileSave, saveActiveFileEdits, prepareFileEditorTextRestore, finishFileEditorTextRestore, discardActiveFileEdits, isFileUnsavedPromptPending, fileUnsavedPromptPlan, beginFileUnsavedPrompt, resolveFileUnsavedPrompt, applyPlainTextFallbackState, maybeHandleUnsavedFileChanges, handleFileUnsavedSaveChoice, handleFileUnsavedDiscardChoice, handleFileUnsavedCancelChoice, setFileViewModeWithGuard, requestHideFileViewer, openFilePathWithGuard, openFilePathWithResolvedMode, openDraftFilePathWithGuard, openDraftFilePath, finalizeFileOpenSuccess, clearFileTouchSelectionState, currentFileTouchSelectMode, currentFileTouchToolbarState, resetFileTouchSelectionState, toggleFileTouchSelectionMode, handleFileTouchMoveButtonPress, moveFileTouchSelection, handleFileTouchSelectionKeydown, handleFileEditorDeleteKeydown, suppressFileEditorNativeDelete, insertIntoActiveFileEditor, pasteFromClipboardIntoActiveFile, handleFilePasteInsert, copyActiveFileSelection, handleFileDiffModeButtonPress, handleFilePreviewModeButtonPress, handleFileEditButtonPress, handleFileEditorSaveShortcut, handleFileVideoPreviewButtonPress, activeFileDownloadApiPath, openFilePath, applyDraftFileLoad, renderFileOpenError, renderDraftFileOpenError, fetchFileOpenResult, isSaveConflictCurrent, reloadSaveConflict, keepEditingSaveConflict, currentSaveConflict, renderSaveConflict } = operations;
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
      currentFileCandidateGitStateMessage,
      setFileCandidateGitStateMessage,
      clearFileCandidateGitStateMessage,
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
      disposeActivePdfRender,
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
  window.CodoxearFileViewerController = Object.freeze({ createFileViewerController });
})();
