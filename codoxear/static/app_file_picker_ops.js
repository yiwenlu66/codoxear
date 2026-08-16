
/* File-picker menu, search, entry identity, and selection operations. */
  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`file picker dependency missing: ${name}`);
    return value;
  }
  function createFilePickerOpsController(options = {}) {
    const { wiring, codoxearFilePicker, normalizeLineNumber, filePickerField, filePickerMenu, filePickerInput,
      api, document, el, getSelected, blockUnavailableFileAction, currentFileViewerSessionId, fileViewerController,
      fileCandidateKey, currentActiveFileDraft, activeFilePathValue, normalizeFileApiPath, renderFilePickerMenu,
      applyFileMenuState, normalizeDraftFilePath, filePickerSectionLabel, duplicateFilePickerPaths,
      rawByteDuplicatePaths, filePickerIdentityHint, filePickerTitle, currentActiveFileIdentity,
      openDraftFilePathWithGuard, openFilePathWithResolvedMode, filePickerSelectionLine,
      ensureCurrentFileViewerSession, resetFilePickerInput, closeFilePickerMenu, resetFileSearchState,
      setFileStatus, requestAnimationFrame } = options;
    requireObject(wiring, "wiring");
    requireObject(codoxearFilePicker, "codoxearFilePicker");
    const menuState = codoxearFilePicker.createMenuState(wiring.createMenuStateOptions({ normalizeLineNumber }));
    const domRuntime = codoxearFilePicker.createMenuDomRuntime(wiring.createMenuDomOptions({
      field: filePickerField, menu: filePickerMenu, input: filePickerInput, menuState,
    }));
    const searchState = codoxearFilePicker.createSearchState(wiring.createSearchStateOptions({
      blocked: () => blockUnavailableFileAction(),
      currentSessionId: () => currentFileViewerSessionId() || getSelected() || "",
      api, inputValue: () => filePickerInput.value, isMenuOpen: () => menuState.isOpen(),
      renderMenu: () => renderFilePickerMenu(), applyMenuState: () => applyFileMenuState(),
      normalizeFileApiPath: (value) => normalizeFileApiPath(value),
    }));
    const entryRuntime = codoxearFilePicker.createEntryRuntime(wiring.createEntryOptions({
      menuState, inputValue: () => filePickerInput.value, candidateKeys: () => fileViewerController().currentFileCandidateKeys(),
      entryForKey: (key) => fileViewerController().fileEntryForKey(key),
      pickerEntryForKey: (key, options) => fileViewerController().pickerEntryForKey(key, options),
      pickerEntryForPath: (path, options) => fileViewerController().pickerEntryForPath(path, options),
      keyForPath: (path, gitPath, apiPath) => fileCandidateKey(path, gitPath, apiPath),
      activeFileDraft: () => currentActiveFileDraft(), activeFilePath: () => activeFilePathValue(),
      searchSnapshot: () => searchState.snapshot(), normalizeFileApiPath: (value) => normalizeFileApiPath(value),
    }));
    const renderRuntime = codoxearFilePicker.createMenuRenderRuntime(wiring.createMenuRenderOptions({
      menu: filePickerMenu, menuState, inputValue: () => filePickerInput.value, visibleEntries: () => entryRuntime.visibleEntries(),
      searchSnapshot: () => searchState.snapshot(), normalizeDraftFilePath: (query) => normalizeDraftFilePath(query),
      draftSuppressed: () => searchState.draftSuppressed(filePickerInput.value), draftEntry: (path) => entryRuntime.draftEntry(path),
      syncActiveDescendant: (focusIndex) => domRuntime.syncActiveDescendant(focusIndex), sectionLabel: (source) => filePickerSectionLabel(source),
      duplicatePaths: (entries) => duplicateFilePickerPaths(entries), rawByteDuplicatePaths: (entries) => rawByteDuplicatePaths(entries),
      identityHint: (entry, duplicatePaths, options) => filePickerIdentityHint(entry, duplicatePaths, options),
      titleForEntry: (entry, hint) => filePickerTitle(entry, hint), normalizeFileApiPath: (value) => normalizeFileApiPath(value),
      activeIdentity: () => currentActiveFileIdentity(), gitStatusMessage: () => fileViewerController().currentFileCandidateGitStateMessage(),
      openDraftFilePath: (draftPath) => openDraftFilePathWithGuard(draftPath),
      openEntry: async (selectedEntry) => {
        try { await openFilePathWithResolvedMode(selectedEntry.path, { line: filePickerSelectionLine(), changed: Boolean(selectedEntry.changed), gitPath: Boolean(selectedEntry.gitPath), apiPath: selectedEntry.apiPath }); }
        catch (error) { setFileStatus(`error: ${error && error.message ? error.message : "unable to inspect path"}`); }
      }, el, createTextNode: (value) => document.createTextNode(value),
    }));
    const inputRuntime = codoxearFilePicker.createInputRuntime(wiring.createInputOptions({
      input: filePickerInput, menuState, ensureCurrentSession: () => ensureCurrentFileViewerSession(),
      renderMenu: () => renderFilePickerMenu(), applyMenuState: () => applyFileMenuState(),
      resetInput: () => resetFilePickerInput(), closeMenu: (opts) => closeFilePickerMenu(opts),
      currentSessionId: () => currentFileViewerSessionId(), selectedSessionId: () => getSelected(),
      resetSearchState: () => resetFileSearchState(), setSearchSessionId: (sid) => searchState.setSessionId(sid),
      scheduleSearch: (query) => searchState.schedule(query), selectionLine: () => filePickerSelectionLine(),
      openDraftFilePathWithGuard: (path) => openDraftFilePathWithGuard(path),
      openFilePathWithResolvedMode: (path, opts) => openFilePathWithResolvedMode(path, opts),
      setStatus: (status) => setFileStatus(status), optionElementById: (id) => document.getElementById(id),
      isFocusInsideField: () => filePickerField.contains(document.activeElement), requestAnimationFrame: requestAnimationFrame,
    }));
    return Object.freeze({ menuState, domRuntime, searchState, entryRuntime, renderRuntime, inputRuntime });
  }

  function bindFilePickerInteractions(options = {}) {
    const { eventBindings, fileBtn, showFileViewer, filePickerInput, filePickerInputRuntime,
      fileModeDiffBtn, fileModePreviewBtn, fileEditBtn, handleFileDiffModeButtonPress,
      handleFilePreviewModeButtonPress, handleFileEditButtonPress, fileVideoPreviewBtn,
      fileVideoPreviewRuntime, fileDownloadBtn, fileDownloadRuntime, activeFileDownloadApiPath,
      codoxearFileViewer, fileTouchSelectBtn, fileTouchCopyBtn, fileTouchPasteBtn, fileTouchUpBtn,
      fileTouchLeftBtn, fileTouchDownBtn, fileTouchRightBtn, toggleFileTouchSelectionMode,
      copyActiveFileSelection, pasteFromClipboardIntoActiveFile, handleFileTouchMoveButtonPress,
      fileCloseBtn, fileBackdrop, requestHideFileViewer, $, fileUnsavedController, fileUnsavedBackdrop,
      filePasteInput, handleFilePasteInsert, hideFilePasteDialog, filePasteBackdrop, chatInner,
      codeBlockCopyRuntime, fileReferenceRuntime, fileDiff, addAppEvent, document, Element,
      isFileViewerOpen, menuState, closeFilePickerMenu } = options;
eventBindings.on(fileBtn, 'click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  void showFileViewer();
});
eventBindings.on(filePickerInput, 'focus', () => filePickerInputRuntime.focus());
eventBindings.on(filePickerInput, 'click', (e) => filePickerInputRuntime.click(e));
eventBindings.on(filePickerInput, 'input', () => filePickerInputRuntime.input());
eventBindings.on(filePickerInput, 'blur', () => filePickerInputRuntime.blur());
eventBindings.on(filePickerInput, 'keydown', (e) => filePickerInputRuntime.keydown(e));
eventBindings.on(fileModeDiffBtn, 'click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  void handleFileDiffModeButtonPress();
});
eventBindings.on(fileModePreviewBtn, 'click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  void handleFilePreviewModeButtonPress();
});
eventBindings.on(fileEditBtn, 'click', async (e) => {
  e.preventDefault();
  e.stopPropagation();
  await handleFileEditButtonPress();
});
eventBindings.on(fileVideoPreviewBtn, 'click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  void fileVideoPreviewRuntime.handleButtonPress();
});

eventBindings.on(fileDownloadBtn, 'click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  fileDownloadRuntime.download(activeFileDownloadApiPath());
});
codoxearFileViewer.bindFileTouchPress(fileTouchSelectBtn, () => {
  toggleFileTouchSelectionMode();
});
codoxearFileViewer.bindFileTouchClick(fileTouchCopyBtn, () => {
  void copyActiveFileSelection();
});
codoxearFileViewer.bindFileTouchClick(fileTouchPasteBtn, () => {
  void pasteFromClipboardIntoActiveFile();
});
codoxearFileViewer.bindFileTouchPress(fileTouchUpBtn, () => {
  handleFileTouchMoveButtonPress("up");
});
codoxearFileViewer.bindFileTouchPress(fileTouchLeftBtn, () => {
  handleFileTouchMoveButtonPress("left");
});
codoxearFileViewer.bindFileTouchPress(fileTouchDownBtn, () => {
  handleFileTouchMoveButtonPress("down");
});
codoxearFileViewer.bindFileTouchPress(fileTouchRightBtn, () => {
  handleFileTouchMoveButtonPress("right");
});
eventBindings.on(fileCloseBtn, 'click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  void requestHideFileViewer();
});
eventBindings.on(fileBackdrop, 'click', () => void requestHideFileViewer());
eventBindings.on($("#fileUnsavedSaveBtn"), 'click', () => fileUnsavedController.handleFileUnsavedSaveChoice());
eventBindings.on($("#fileUnsavedDiscardBtn"), 'click', () => fileUnsavedController.handleFileUnsavedDiscardChoice());
eventBindings.on($("#fileUnsavedCancelBtn"), 'click', () => fileUnsavedController.handleFileUnsavedCancelChoice());
eventBindings.on(fileUnsavedBackdrop, 'click', () => fileUnsavedController.handleFileUnsavedCancelChoice());
eventBindings.on($("#filePasteInsertBtn"), 'click', () => {
  handleFilePasteInsert(filePasteInput.value);
});
eventBindings.on($("#filePasteCancelBtn"), 'click', () => hideFilePasteDialog({ restoreFocus: true }));
eventBindings.on(filePasteBackdrop, 'click', () => hideFilePasteDialog({ restoreFocus: true }));
chatInner.addEventListener("click", (e) => {
  if (codeBlockCopyRuntime.handleClick(e)) return;
  void fileReferenceRuntime.handleClick(e);
});
fileDiff.addEventListener("click", (e) => {
  void fileReferenceRuntime.handleClick(e);
});
addAppEvent(document, "click", (e) => {
  const t = e.target instanceof Element ? e.target : null;
  if (!t) return;
  if (isFileViewerOpen() && filePickerMenuState.isOpen() && !t.closest("#fileCandRow")) {
    closeFilePickerMenu({ restoreInput: true });
  }
});
  }

  function createFilePickerOperationDelegates(options = {}) {
    const { fileViewerController, fileModeControlsRuntime, filePickerDomRuntime, filePickerMenuState,
      filePickerInput, filePickerInputRuntime, activeFilePathValue, openedFileRuntime, fileReferenceRuntime,
      filePickerSearchState, filePickerRenderRuntime, fileViewerPanelRuntime, getSelected, getSessionIndex,
      stripPathLocationSuffix } = options;
    return Object.freeze({
      openDraftFilePathWithGuard: async (path) => fileViewerController.openDraftFilePathWithGuard(path),
      requestHideFileViewer: async () => fileViewerController.requestHideFileViewer(),
      handleFileDiffModeButtonPress: async () => fileViewerController.handleFileDiffModeButtonPress(),
      handleFilePreviewModeButtonPress: async () => fileViewerController.handleFilePreviewModeButtonPress(),
      handleFileEditButtonPress: async () => fileViewerController.handleFileEditButtonPress(),
      activeFileDownloadApiPath: () => fileViewerController.activeFileDownloadApiPath(),
      setFileViewMode: (mode) => fileViewerController.setFileViewMode(mode),
      applyFileMode: () => fileModeControlsRuntime.apply(fileViewerController.currentFileModeControlState()),
      applyFileMenuState: () => filePickerDomRuntime.apply(),
      resetFilePickerInput: () => filePickerDomRuntime.resetInput(activeFilePathValue() || ""),
      closeFilePickerMenu: ({ restoreInput = false } = {}) => filePickerDomRuntime.close({ restoreInput, inputValue: activeFilePathValue() || "" }),
      filePickerSelectionLine: () => filePickerMenuState.selectionLine(filePickerInput.value),
      openFilePickerSearchQuery: (query, opts = {}) => filePickerInputRuntime.openSearchQuery(query, opts),
      normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
      setFilePath: (rel, opts = {}) => fileViewerPanelRuntime.setFilePath(rel, opts),
      fileCandidateKey: (path, gitPath = false, apiPath = "") => fileViewerController.fileCandidateKey(path, gitPath, apiPath),
      fileEntryForPath: (path, gitPath = false, apiPath = "") => fileViewerController.fileEntryForPath(path, gitPath, apiPath),
      openFilePathWithResolvedMode: async (path, opts = {}) => fileViewerController.openFilePathWithResolvedMode(path, opts),
      upsertFileEntry: (entry) => fileViewerController.upsertFileEntry(entry),
      rememberOpenedFile: (relPath, absPath = null) => openedFileRuntime.remember(relPath, absPath),
      collectMessageFileRefs: () => fileReferenceRuntime.collectMessageFileRefs(),
      resetFileSearchState: () => filePickerSearchState.reset(),
      filePickerSearchSnapshot: () => filePickerSearchState.snapshot(),
      renderFilePickerMenu: () => filePickerRenderRuntime.render(),
      upgradeCandidateFileRefs: async (root) => fileReferenceRuntime.upgradeCandidateRefs(root),
      sessionRelativePath: (rawPath, sidOverride = null) => {
        const sid = typeof sidOverride === "string" && sidOverride ? sidOverride : getSelected();
        const session = sid ? getSessionIndex().get(sid) : null;
        if (!session || !session.cwd) return null;
        const abs = stripPathLocationSuffix(rawPath), cwd = String(session.cwd || "").replace(/\/+$/, "");
        if (!abs) return null; if (abs === cwd) return ".";
        return abs.startsWith(cwd + "/") ? abs.slice(cwd.length + 1) : null;
      },
    });
  }

export { createFilePickerOpsController, createFilePickerOperationDelegates, bindFilePickerInteractions };
