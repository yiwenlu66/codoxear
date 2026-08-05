/* File operations composition: viewer, editor, picker, unsaved-change, and touch keyboard. */
(function installCodoxearFileOps(global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file operations dependency missing: ${name}`);
    return value;
  }

  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`file operations dependency missing: ${name}`);
    return value;
  }

  function createFileOpsController(options = {}) {
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const getSessionIndex = requireFunction(options.getSessionIndex, "getSessionIndex");
    const getSessionLifecycleController = requireFunction(options.getSessionLifecycleController, "getSessionLifecycleController");
    const {
      wiring, document, window, HTMLElement, requestAnimationFrame, setTimeout,
      $, el, iconSvg, resolveAppUrl, api, setToast, confirmApp, addAppEvent,
      sessionLaunchFailed, normalizeLineNumber, markdownPreviewHtml,
      blockedFileMessage, listFromFilesField, listFromFileRecords, baseName,
      codoxearFilePicker, codoxearFileViewer, codoxearFileEditor, codoxearFileEditMode,
      codoxearFileTouch, codoxearFileViewerIntegration, codoxearDialogMenus,
      prepareModalOpen, afterModalVisibilityChanged, focusModalCloseButton, restoreModalFocus,
      isModalTargetOpen, newSessionDialogController, sessionEditController, eventBindings,
      codoxearFileHelpers, copyToClipboard, dialogMenuController, duplicateFilePickerPaths,
      editCloseBtn, editDependencyBtn, editDependencyMenu, editNameInput, editPriorityRange,
      editPriorityResetBtn, editPriorityValue, editSaveBtn, editSnoozeCustomDate,
      editSnoozeCustomRow, editSnoozeCustomTime, editSnoozeModeButtons, editStatus, editViewer,
      fileBtn, filePickerIdentityHint, filePickerSectionLabel, filePickerTitle, fmtBytes,
      formatPriorityOffset, handleAppAuthLoss, isDiffableFileKind, isMarkdownPreviewable,
      isTextEntryElement, isTextFileKind, modalIsolationTargets, normalizeDraftFilePath,
      parseLocalFileRef, rawByteDuplicatePaths, refreshSessions, selectedSessionLaunchFailed,
      sessionDisplayName, sessionTitleWithId, setPickerButtonContent, storageGetItem,
      storageSetItem, stripPathLocationSuffix, titleLabel, useTouchFileEditorControls,
      filePickerField, filePickerMenu, filePickerInput, fileStatus, fileDiff, fileImage,
      fileVideo, fileVideoPreviewBtn, fileTouchToolbar, fileTouchActions, fileTouchDpad,
      fileTouchCopyBtn, fileTouchPasteBtn, fileTouchSelectBtn, fileTouchUpBtn, fileTouchLeftBtn,
      fileTouchDownBtn, fileTouchRightBtn, fileModeDiffBtn, fileModePreviewBtn, fileDownloadBtn,
      fileBackdrop, fileViewer, fileCloseBtn, fileUnsavedBackdrop, fileUnsavedDialog,
      filePasteBackdrop, filePasteDialog, filePasteInput, fileEditBtn, chatInner,
      codeBlockCopyRuntime, appConfirm, appConfirmFocusableControls, resolveAppConfirm,
      sendChoice, closeSendChoiceDialog, queueViewer, hideQueueViewer, helpViewer,
      hideHelpViewer, diagViewer, hideDiagViewer, voiceController, hideVoiceSettingsDialog
    } = options;
    requireObject(wiring, "wiring");
    requireObject(codoxearFilePicker, "codoxearFilePicker");
    requireObject(codoxearFileViewer, "codoxearFileViewer");
    requireObject(codoxearFileEditor, "codoxearFileEditor");

const dialogMenusController = codoxearDialogMenus.createDialogMenusController(wiring.createDialogMenusOptions({
  sessionEditController: () => sessionEditController,
  newSessionDialogController: () => newSessionDialogController,
}));

const FILE_CANDIDATE_CACHE_TTL_MS = 15000;
const filePickerMenuState = codoxearFilePicker.createMenuState(wiring.createMenuStateOptions({
  normalizeLineNumber,
}));
const filePickerDomRuntime = codoxearFilePicker.createMenuDomRuntime(wiring.createMenuDomOptions({
  field: filePickerField,
  menu: filePickerMenu,
  input: filePickerInput,
  menuState: filePickerMenuState,
}));
const filePickerSearchState = codoxearFilePicker.createSearchState(wiring.createSearchStateOptions({
  blocked: () => blockUnavailableFileAction(),
  currentSessionId: () => currentFileViewerSessionId() || getSelected() || "",
  api,
  inputValue: () => filePickerInput.value,
  isMenuOpen: () => filePickerMenuState.isOpen(),
  renderMenu: () => renderFilePickerMenu(),
  applyMenuState: () => applyFileMenuState(),
  normalizeFileApiPath: (value) => normalizeFileApiPath(value),
}));
const filePickerEntryRuntime = codoxearFilePicker.createEntryRuntime(wiring.createEntryOptions({
  menuState: filePickerMenuState,
  inputValue: () => filePickerInput.value,
  candidateKeys: () => fileViewerController.currentFileCandidateKeys(),
  entryForKey: (key) => fileViewerController.fileEntryForKey(key),
  pickerEntryForKey: (key, options) => fileViewerController.pickerEntryForKey(key, options),
  pickerEntryForPath: (path, options) => fileViewerController.pickerEntryForPath(path, options),
  keyForPath: (path, gitPath, apiPath) => fileCandidateKey(path, gitPath, apiPath),
  activeFileDraft: () => currentActiveFileDraft(),
  activeFilePath: () => activeFilePathValue(),
  searchSnapshot: () => filePickerSearchSnapshot(),
  normalizeFileApiPath: (value) => normalizeFileApiPath(value),
}));
const filePickerRenderRuntime = codoxearFilePicker.createMenuRenderRuntime(wiring.createMenuRenderOptions({
  menu: filePickerMenu,
  menuState: filePickerMenuState,
  inputValue: () => filePickerInput.value,
  visibleEntries: () => filePickerEntryRuntime.visibleEntries(),
  searchSnapshot: () => filePickerSearchSnapshot(),
  normalizeDraftFilePath: (query) => normalizeDraftFilePath(query),
  draftSuppressed: () => filePickerSearchState.draftSuppressed(filePickerInput.value),
  draftEntry: (path) => filePickerEntryRuntime.draftEntry(path),
  syncActiveDescendant: (focusIndex) => filePickerDomRuntime.syncActiveDescendant(focusIndex),
  sectionLabel: (source) => filePickerSectionLabel(source),
  duplicatePaths: (entries) => duplicateFilePickerPaths(entries),
  rawByteDuplicatePaths: (entries) => rawByteDuplicatePaths(entries),
  identityHint: (entry, duplicatePaths, options) => filePickerIdentityHint(entry, duplicatePaths, options),
  titleForEntry: (entry, hint) => filePickerTitle(entry, hint),
  normalizeFileApiPath: (value) => normalizeFileApiPath(value),
  activeIdentity: () => currentActiveFileIdentity(),
  gitStatusMessage: () => fileViewerController.currentFileCandidateGitStateMessage(),
  openDraftFilePath: (draftPath) => openDraftFilePathWithGuard(draftPath),
  openEntry: async (selectedEntry) => {
    try {
      await openFilePathWithResolvedMode(selectedEntry.path, { line: filePickerSelectionLine(), changed: Boolean(selectedEntry.changed), gitPath: Boolean(selectedEntry.gitPath), apiPath: selectedEntry.apiPath });
    } catch (e) {
      fileStatus.textContent = `error: ${e && e.message ? e.message : "unable to inspect path"}`;
    }
  },
  el,
  createTextNode: (value) => document.createTextNode(value),
}));
const filePickerInputRuntime = codoxearFilePicker.createInputRuntime(wiring.createInputOptions({
  input: filePickerInput,
  menuState: filePickerMenuState,
  ensureCurrentSession: () => ensureCurrentFileViewerSession(),
  renderMenu: () => renderFilePickerMenu(),
  applyMenuState: () => applyFileMenuState(),
  resetInput: () => resetFilePickerInput(),
  closeMenu: (options) => closeFilePickerMenu(options),
  currentSessionId: () => currentFileViewerSessionId(),
  selectedSessionId: () => getSelected(),
  resetSearchState: () => resetFileSearchState(),
  setSearchSessionId: (sessionId) => filePickerSearchState.setSessionId(sessionId),
  scheduleSearch: (query) => filePickerSearchState.schedule(query),
  selectionLine: () => filePickerSelectionLine(),
  openDraftFilePathWithGuard: (path) => openDraftFilePathWithGuard(path),
  openFilePathWithResolvedMode: (path, options) => openFilePathWithResolvedMode(path, options),
  setStatus: (status) => {
    fileStatus.textContent = status;
  },
  optionElementById: (id) => document.getElementById(id),
  isFocusInsideField: () => filePickerField.contains(document.activeElement),
  requestAnimationFrame: (callback) => requestAnimationFrame(callback),
}));
const MONACO_LOADER_TIMEOUT_MS = 4000;
const PDFJS_LOADER_TIMEOUT_MS = 6000;
const fileEditorRuntime = codoxearFileEditor.createFileEditorRuntime();
const fileEditorMonacoLoader = codoxearFileEditor.createMonacoLoader(wiring.createMonacoLoaderOptions({
  resolveAppUrl,
  timeoutMs: MONACO_LOADER_TIMEOUT_MS,
}));
const fileEditorRenderer = codoxearFileEditor.createFileEditorRenderer(wiring.createFileEditorRendererOptions({
  runtime: fileEditorRuntime,
  monacoLoader: fileEditorMonacoLoader,
  host: fileDiff,
  normalizeLineNumber,
  requestAnimationFrame: (callback) => requestAnimationFrame(callback),
  setTimeout: (callback, delay) => setTimeout(callback, delay),
  isCurrentFileOpenRequest: (request) => isCurrentFileOpenRequest(request),
  renderPlainTextFallback: (rel, text, lineNumber, reason) => renderPlainTextFallback(rel, text, lineNumber, reason),
  disposeFileEditor: () => disposeFileEditor(),
  currentEditorKind: () => currentFileEditorKind(),
  setEditorKind: (kind) => setFileEditorKind(kind),
  currentFileEditMode: () => currentFileEditMode(),
  currentActiveFileEditable: () => currentActiveFileEditable(),
  isUnavailable: () => isFileViewerSessionUnavailable(),
  isProgrammaticChange: () => fileViewerController.isFileEditorProgrammaticChange(),
  currentTouchSelectMode: () => currentFileTouchSelectMode(),
  resetTouchSelectionState: () => resetFileTouchSelectionState(),
  currentActiveFileText: () => currentActiveFileText(),
  setDirty: (dirty) => setFileDirty(dirty),
  runProgrammaticChange: (callback) => fileViewerController.runFileEditorProgrammaticChange(callback),
  syncReadOnly: () => syncFileEditorReadOnly(),
  updateTouchToolbar: () => updateFileTouchToolbar(),
}));
const filePdfLoader = codoxearFileViewer.createPdfLoader(wiring.createPdfLoaderOptions({
  resolveAppUrl,
  timeoutMs: PDFJS_LOADER_TIMEOUT_MS,
}));
const fileFallbackRuntime = codoxearFileViewer.createFileFallbackRuntime(wiring.createFileFallbackOptions({
  host: fileDiff,
  el,
  normalizeLineNumber,
  requestAnimationFrame: (callback) => requestAnimationFrame(callback),
  disposeFileEditor: () => disposeFileEditor(),
  disposePdfRender: () => disposePdfRender(),
  clearFileVideo: () => clearFileVideo(),
  setFileRenderSurface: (surface) => setFileRenderSurface(surface),
  setFileEditorKind: (kind) => setFileEditorKind(kind),
  applyPlainTextFallbackState: () => fileViewerController.applyPlainTextFallbackState(),
  updateFileTouchToolbar: () => updateFileTouchToolbar(),
  currentSessionId: () => currentFileViewerSessionId() || getSelected() || "",
  markdownPreviewHtml: (body, context) => markdownPreviewHtml(body, context),
  upgradeCandidateFileRefs: (node) => upgradeCandidateFileRefs(node),
  blockedFileMessage: (rel, reason, viewerMaxBytes, size) => blockedFileMessage(rel, reason, viewerMaxBytes, size),
}));
const fileDownloadRuntime = codoxearFileViewer.createFileDownloadRuntime(wiring.createFileDownloadOptions({
  resolveAppUrl,
  document,
}));
const filePdfRenderRuntime = codoxearFileViewer.createFilePdfRenderRuntime(wiring.createFilePdfRenderOptions({
  host: fileDiff,
  el,
  ensurePdfJs: () => ensurePdfJs(),
  createCanvas: () => document.createElement("canvas"),
  devicePixelRatio: () => window.devicePixelRatio || 1,
  disposeFileEditor: () => disposeFileEditor(),
  disposePdfRender: () => disposePdfRender(),
  clearFileVideo: () => clearFileVideo(),
  setFileRenderSurface: (surface) => setFileRenderSurface(surface),
  renderDownloadFallback: (rel, url, reason) => renderDownloadFallback(rel, url, reason),
  isCurrentFileOpenRequest: (request) => isCurrentFileOpenRequest(request),
  setActivePdfRenderState: (state) => fileViewerController.setActivePdfRenderState(state),
  isActivePdfRenderState: (state) => fileViewerController.isActivePdfRenderState(state),
  updateFileTouchToolbar: () => updateFileTouchToolbar(),
  IntersectionObserverCtor: typeof IntersectionObserver === "function" ? IntersectionObserver : null,
}));
const filePasteDialogRuntime = codoxearFileViewer.createFilePasteDialogRuntime(wiring.createFilePasteDialogOptions({
  backdrop: filePasteBackdrop,
  dialog: filePasteDialog,
  input: filePasteInput,
  prepareModalOpen,
  afterModalVisibilityChanged,
  focusActiveEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind()),
  requestAnimationFrame: (callback) => requestAnimationFrame(callback),
}));
const fileRenderSurfaceRuntime = codoxearFileViewer.createFileRenderSurfaceRuntime(wiring.createFileRenderSurfaceOptions({
  diff: fileDiff,
  image: fileImage,
  video: fileVideo,
  videoPreviewButton: fileVideoPreviewBtn,
  clearActiveVideoFallback: () => fileViewerController.clearActiveVideoFallback(),
}));
const fileModeControlsRuntime = codoxearFileViewer.createFileModeControlsRuntime(wiring.createFileModeControlsOptions({
  diffButton: fileModeDiffBtn,
  previewButton: fileModePreviewBtn,
  downloadButton: fileDownloadBtn,
  videoPreviewButton: fileVideoPreviewBtn,
  hideFilePasteDialog: () => hideFilePasteDialog(),
  setFileEditMode: (mode) => fileEditModeController.setFileEditMode(mode),
  syncFileEditorReadOnly: () => syncFileEditorReadOnly(),
  updateFileEditButton: () => updateFileEditButton(),
}));
const fileTouchToolbarRuntime = codoxearFileViewer.createFileTouchToolbarRuntime(wiring.createFileTouchToolbarOptions({
  toolbar: fileTouchToolbar,
  actions: fileTouchActions,
  dpad: fileTouchDpad,
  copyButton: fileTouchCopyBtn,
  pasteButton: fileTouchPasteBtn,
  selectButton: fileTouchSelectBtn,
}));
const fileViewerModalRuntime = codoxearFileViewer.createFileViewerModalRuntime(wiring.createFileViewerModalOptions({
  backdrop: fileBackdrop,
  viewer: fileViewer,
  pickerInput: filePickerInput,
  closeButton: fileCloseBtn,
  prepareModalOpen,
  afterModalVisibilityChanged,
  focusModalCloseButton,
  restoreModalFocus,
  isModalTargetOpen,
  setReturnFocusElement: (element, ElementCtor) => fileViewerController.setFileViewerReturnFocusElement(element, ElementCtor),
  takeReturnFocusElement: () => fileViewerController.takeFileViewerReturnFocusElement(),
}));
const fileUnsavedDialogRuntime = codoxearFileViewer.createFileUnsavedDialogRuntime(wiring.createFileUnsavedDialogOptions({
  backdrop: fileUnsavedBackdrop,
  dialog: fileUnsavedDialog,
  viewer: fileViewer,
  title: fileUnsavedDialog.querySelector(".title"),
  message: fileUnsavedDialog.querySelector(".muted"),
  saveButton: $("#fileUnsavedSaveBtn"),
  discardButton: $("#fileUnsavedDiscardBtn"),
  cancelButton: $("#fileUnsavedCancelBtn"),
  prepareModalOpen,
  afterModalVisibilityChanged,
  restoreModalFocus,
  isModalTargetOpen,
  requestAnimationFrame: (callback) => requestAnimationFrame(callback),
  promptPlan: () => fileViewerController.fileUnsavedPromptPlan(),
  beginPrompt: () => fileViewerController.beginFileUnsavedPrompt(),
  resolvePrompt: (choice) => fileViewerController.resolveFileUnsavedPrompt(choice),
  setReturnFocusElement: (element, ElementCtor) => fileViewerController.setFileUnsavedReturnFocusElement(element, ElementCtor),
  takeReturnFocusElement: () => fileViewerController.takeFileUnsavedReturnFocusElement(),
  isUnavailable: () => isFileViewerSessionUnavailable(),
}));
const codoxearFileUnsaved = window.CodoxearFileUnsaved;
if (!codoxearFileUnsaved || typeof codoxearFileUnsaved.createFileUnsavedController !== "function")
  throw new Error("Codoxear file unsaved controller failed to load");
const fileUnsavedController = codoxearFileUnsaved.createFileUnsavedController(wiring.createFileUnsavedOptions({
  documentTarget: document,
  ElementCtor: HTMLElement,
  dialogRuntime: fileUnsavedDialogRuntime,
  getFileViewerController: () => fileViewerController,
}));

function currentFileViewerSessionId() {
  return fileViewerController.currentFileViewerSessionId();
}

function currentFileSessionId() {
  return String(currentFileViewerSessionId() || getSelected() || "").trim();
}

function isFileViewerSessionUnavailable() {
  return fileViewerController.isFileViewerSessionUnavailable();
}

function blockUnavailableFileAction() {
  return fileViewerController.blockUnavailableFileAction();
}

function currentActiveFileIdentity() {
  return fileViewerController.currentActiveFileIdentity();
}

function activeFilePathValue() {
  return currentActiveFileIdentity().path;
}

function currentFileEditorKind() {
  return fileViewerController.currentFileEditorKind();
}

function setFileEditorKind(kind) {
  return fileViewerController.setFileEditorKind(kind);
}

function isCurrentFileOpenRequest(request) {
  return fileViewerController.isCurrentFileOpenRequest(request);
}

function clearFileVideo() {
  return fileRenderSurfaceRuntime.clearVideo();
}

function setFileRenderSurface(surface) {
  return fileRenderSurfaceRuntime.setSurface(surface);
}

function resetFileViewerPanel() {
  return fileViewerPanelRuntime.resetPanel();
}

function renderEmptyFileViewerTarget({ updateTouchToolbar = false } = {}) {
  return fileViewerPanelRuntime.renderEmptyTarget({ updateTouchToolbar });
}

async function ensureCurrentFileViewerSession() {
  return await fileViewerLifecycleRuntime.ensureCurrentSession();
}

function disposeFileEditor() {
  return fileEditorRuntime.disposeCurrentFile({
    finishProgrammaticChange: () => fileViewerController.finishFileEditorProgrammaticChange(),
    clearHost: () => {
      fileDiff.innerHTML = "";
    },
    setFileEditorKind: (kind) => setFileEditorKind(kind),
    clearFileTouchSelectionState: () => clearFileTouchSelectionState(),
  });
}

function disposePdfRender() {
  return fileViewerController.disposeActivePdfRender();
}

function isFileViewerOpen() {
  return fileViewerModalRuntime.isOpen();
}

function syncFileEditorReadOnly() {
  return fileViewerController.syncFileEditorReadOnly();
}

function updateFileTouchToolbar() {
  return fileTouchToolbarRuntime.update(fileViewerController.currentFileTouchToolbarState());
}

function clearFileTouchSelectionState() {
  return fileViewerController.clearFileTouchSelectionState();
}

function currentFileTouchSelectMode() {
  return fileViewerController.currentFileTouchSelectMode();
}

function resetFileTouchSelectionState(options) {
  return fileViewerController.resetFileTouchSelectionState(options);
}

function toggleFileTouchSelectionMode() {
  return fileViewerController.toggleFileTouchSelectionMode();
}

function handleFileTouchMoveButtonPress(direction) {
  return fileViewerController.handleFileTouchMoveButtonPress(direction);
}

function handleFileEditorSaveShortcut(e) {
  return fileViewerController.handleFileEditorSaveShortcut(e);
}

function handleFileEditorDeleteKeydown(e) {
  return fileViewerController.handleFileEditorDeleteKeydown(e);
}

function suppressFileEditorNativeDelete(e) {
  return fileViewerController.suppressFileEditorNativeDelete(e);
}

async function copyActiveFileSelection() {
  return await fileViewerController.copyActiveFileSelection();
}

function hideFilePasteDialog({ restoreFocus = false } = {}) {
  return filePasteDialogRuntime.hide({ restoreFocus });
}

function showFilePasteDialog() {
  return filePasteDialogRuntime.show();
}

async function pasteFromClipboardIntoActiveFile() {
  return await fileViewerController.pasteFromClipboardIntoActiveFile();
}

function handleFilePasteInsert(text) {
  return fileViewerController.handleFilePasteInsert(text);
}

function updateFileEditButton() {
  return fileViewerController.updateFileEditButton();
}

function currentFileDirty() {
  return fileViewerController.currentFileDirty();
}

function setFileDirty(nextDirty) {
  return fileViewerController.setFileDirty(nextDirty);
}

function resetActiveFileBufferState() {
  fileViewerController.resetActiveFileBufferState();
}

function currentActiveFileText() {
  return fileViewerController.currentActiveFileText();
}

function currentActiveFileEditable() {
  return fileViewerController.currentActiveFileEditable();
}

function currentActiveFileDraft() {
  return fileViewerController.currentActiveFileDraft();
}

function getFileEditorText() {
  return fileEditorRuntime.currentFileText(currentFileEditorKind(), currentActiveFileText());
}

function restoreFileEditorText(text) {
  return fileEditorRuntime.restoreCurrentFileText(text, {
    prepareFileEditorTextRestore: (value) => fileViewerController.prepareFileEditorTextRestore(value),
    currentFileEditorKind: () => currentFileEditorKind(),
    runFileEditorProgrammaticChange: (callback) => fileViewerController.runFileEditorProgrammaticChange(callback),
    finishFileEditorTextRestore: () => fileViewerController.finishFileEditorTextRestore(),
  });
}

function renderPlainTextFallback(rel, text, lineNumber = null, reason = "Rich file viewer unavailable") {
  return fileFallbackRuntime.applyPlainText(rel, text, lineNumber, reason);
}

function renderDownloadFallback(rel, url, reason = "Preview unavailable") {
  return fileFallbackRuntime.applyDownload(rel, url, reason);
}

async function ensurePdfJs() {
  return await filePdfLoader.ensure();
}

async function renderMonacoFile(rel, text, lineNumber = null, langOverride = "", request = null) {
  return await fileEditorRenderer.renderFile(rel, text, lineNumber, langOverride, request);
}

async function renderMonacoDiff(rel, originalText, modifiedText, lineNumber = null, request = null) {
  return await fileEditorRenderer.renderDiff(rel, originalText, modifiedText, lineNumber, request);
}

function renderMarkdownPreview(rel, text) {
  return fileFallbackRuntime.applyMarkdown(rel, text);
}

function renderBlockedFileNotice(rel, reason, viewerMaxBytes, size) {
  return fileFallbackRuntime.applyBlocked(rel, reason, viewerMaxBytes, size);
}

async function renderPdfFile(rel, url, request) {
  return await filePdfRenderRuntime.render(rel, url, request);
}

function currentFileEditMode() {
  return fileViewerController.currentFileEditMode();
}

const fileEditModeController = codoxearFileEditMode.createFileEditModeController(wiring.createFileEditModeOptions({
  fileViewerController: () => fileViewerController,
}));

const fileInspectRuntime = codoxearFileViewer.createFileInspectRuntime(wiring.createFileInspectOptions({
  currentSessionId: () => currentFileViewerSessionId(),
  selectedSessionId: () => getSelected(),
  normalizeFileApiPath: (value) => normalizeFileApiPath(value),
  api: (url, options) => api(url, options),
}));

const fileViewerController = codoxearFileViewer.createFileViewerController(wiring.createFileViewerOptions({
  el,
  fileStatus,
  fileEditButton: fileEditBtn,
  iconSvg,
  currentSessionId: () => currentFileViewerSessionId(),
  currentFileSessionId: () => currentFileSessionId(),
  normalizeLineNumber,
  normalizeFileApiPath,
  isFileViewerOpen: () => isFileViewerOpen(),
  hideFileUnsavedDialog: (choice) => fileUnsavedController.hideFileUnsavedDialog(choice),
  resetFileSearchState: () => resetFileSearchState(),
  closeFilePickerMenu: (options) => closeFilePickerMenu(options),
  isTextFileKind: (kind) => isTextFileKind(kind),
  isDiffableFileKind: (kind) => isDiffableFileKind(kind),
  confirmReload: (message) => confirmApp({ title: "Reload file from disk?", message, confirmText: "Reload", cancelText: "Cancel", destructive: true }),
  promptUnsavedFileChoice: () => fileUnsavedController.promptFileUnsavedChoice(),
  restoreFileEditorText: (text) => restoreFileEditorText(text),
  hideFileViewer: () => hideFileViewer(),
  setFilePath: (path, options) => setFilePath(path, options),
  resetFileViewerPanel: () => resetFileViewerPanel(),
  applyFileLoadResult: (rel, result, request, options) => applyFileLoadResult(rel, result, request, options),
  normalizeDraftFilePath: (path) => normalizeDraftFilePath(path),
  inspectSessionFilePath: (path, options) => fileInspectRuntime.inspectSessionFilePath(path, options),
  api: (url, options) => api(url, options),
  focusEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind()),
  disposeOpenRender: () => disposePdfRender(),
  initialFileViewMode: storageGetItem("codexweb.fileViewMode") || "diff",
  initialFileNonDiffMode: storageGetItem("codexweb.fileNonDiffMode") === "preview" ? "preview" : "file",
  persistFileViewMode: (mode) => storageSetItem("codexweb.fileViewMode", mode),
  persistFileNonDiffMode: (mode) => storageSetItem("codexweb.fileNonDiffMode", mode),
  isMarkdownPreviewable,
  resetActiveFileBufferState: () => resetActiveFileBufferState(),
  updateFileTouchToolbar: () => updateFileTouchToolbar(),
  useTouchFileEditorControls: () => useTouchFileEditorControls(),
  hasActiveFileCodeEditor: () => Boolean(fileEditorRuntime.activeCodeEditor(currentFileEditorKind())),
  hasBlockingFileEditorModal: () => modalIsolationTargets.some((node) => node !== fileViewer && isModalTargetOpen(node)),
  isTextEntryTarget: (target) => isTextEntryElement(target),
  eventTargetElement: (value) => value instanceof HTMLElement ? value : null,
  normalizeFileEditorPosition: (editor, position) => fileEditorRuntime.normalizePosition(editor, position),
  applyFileEditorSelection: (editor, cursor, anchor) => fileEditorRuntime.applySelection(editor, cursor, anchor, fileEditorMonacoLoader.selectionCtor()),
  isCollapsedFileSelection: (selection) => fileEditorRuntime.isCollapsedSelection(selection),
  fileEditorEditSupportAvailable: () => fileEditorMonacoLoader.editSupportAvailable(),
  updateFileDiffEditorOptions: (options) => fileEditorRuntime.updateEditorOptions(currentFileEditorKind(), options),
  showFilePasteDialog: () => showFilePasteDialog(),
  hideFilePasteDialog: (options) => hideFilePasteDialog(options),
  clipboardReadAvailable: () => Boolean(window.isSecureContext && navigator.clipboard && typeof navigator.clipboard.readText === "function"),
  readClipboardText: () => navigator.clipboard.readText(),
  isActiveFileEditorInput: (target) => fileEditorRuntime.isActiveInput(currentFileEditorKind(), target, HTMLElement),
  getActiveFileSelectionText: () => fileEditorRuntime.activeSelectionText(currentFileEditorKind()),
  copyToClipboard: (text) => copyToClipboard(text),
  focusActiveFileCodeEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind()),
  nowMs: () => Date.now(),
  setToast: (message) => setToast(message),
  setFileViewMode: (mode) => setFileViewMode(mode),
  renderMonacoFile: (rel, text, lineNumber, langOverride, request) => renderMonacoFile(rel, text, lineNumber, langOverride, request),
  getFileEditorText: () => getFileEditorText(),
  fmtBytes: (value) => fmtBytes(value),
  applyFileMode: () => applyFileMode(),
  rememberOpenedFile: (rel, absPath) => rememberOpenedFile(rel, absPath),
  historyFileSelectionForSession: (sessionId) => openedFileRuntime.historySelection(sessionId),
  renderFilePickerMenu: () => renderFilePickerMenu(),
}));
sessionEditController = window.CodoxearSessionEdit.createSessionEditController(wiring.createSessionEditOptions({
  documentTarget: document,
  ElementCtor: HTMLElement,
  el,
  editCloseBtn,
  editStatus,
  editNameInput,
  editPriorityRange,
  editPriorityValue,
  editPriorityResetBtn,
  editSnoozeModeButtons,
  editSnoozeCustomDate,
  editSnoozeCustomTime,
  editSnoozeCustomRow,
  editDependencyBtn,
  editDependencyMenu,
  editSaveBtn,
  editCancelBtn: $("#editCancelBtn"),
  editViewer,
  getSessionInfo: (sid) => getSessionIndex().get(sid),
  getSessions: () => Array.from(getSessionIndex().values()),
  selectedSessionId: () => getSelected(),
  sessionDisplayName,
  baseName,
  formatPriorityOffset,
  setPickerButtonContent,
  api,
  refreshSessions,
  setToast,
  setTitle: (_sid, session) => { if (session) titleLabel.textContent = sessionTitleWithId(session); },
  prepareModalOpen,
  afterModalVisibilityChanged,
  positionDialogMenu: (menu, anchorBtn) => dialogMenuController.positionDialogMenu(menu, anchorBtn),
  addAppEvent,
}));
const fileViewerPanelRuntime = codoxearFileViewer.createFileViewerPanelRuntime(wiring.createFileViewerPanelOptions({
  controller: fileViewerController,
  disposeFileEditor: () => disposeFileEditor(),
  resetRenderSurface: () => fileRenderSurfaceRuntime.reset(),
  resetFilePickerInput: () => resetFilePickerInput(),
  renderFilePickerMenu: () => renderFilePickerMenu(),
  closeFilePickerMenu: () => closeFilePickerMenu(),
  applyFileMode: () => applyFileMode(),
  updateFileTouchToolbar: () => updateFileTouchToolbar(),
  setStatus: (status) => {
    fileStatus.textContent = status;
  },
}));
const fileViewerLifecycleRuntime = codoxearFileViewer.createFileViewerLifecycleRuntime(wiring.createFileViewerLifecycleOptions({
  controller: fileViewerController,
  beginHide: () => fileViewerModalRuntime.beginHide(),
  hideDisplay: () => fileViewerModalRuntime.hideDisplay(),
  finishHide: (state) => fileViewerModalRuntime.finishHide(state),
  hideFileUnsavedDialog: () => fileUnsavedController.hideFileUnsavedDialog(),
  hideFilePasteDialog: () => hideFilePasteDialog(),
  resetFileViewerPanel: () => resetFileViewerPanel(),
  closeFilePickerMenu: (options) => closeFilePickerMenu(options),
  resetFileSearchState: () => resetFileSearchState(),
  setFileSearchSessionId: (sessionId) => filePickerSearchState.setSessionId(sessionId),
  updateFileTouchToolbar: () => updateFileTouchToolbar(),
  isFileViewerOpen: () => isFileViewerOpen(),
  selectedSessionId: () => getSelected(),
  maybeHandleUnsavedFileChanges: () => fileUnsavedController.maybeHandleUnsavedFileChanges(),
  filePickerSearchSessionId: () => filePickerSearchSnapshot().sessionId,
  refreshFileCandidates: (options) => refreshFileCandidates(options),
  setFilePath: (path, options) => setFilePath(path, options),
  openFilePathWithResolvedMode: (path, options) => openFilePathWithResolvedMode(path, options),
  renderEmptyFileViewerTarget: (options) => renderEmptyFileViewerTarget(options),
  setStatus: (status) => {
    fileStatus.textContent = status;
  },
  showModal: (options) => fileViewerModalRuntime.show({ ...options, activeElement: document.activeElement, ElementCtor: HTMLElement }),
  setFileViewMode: (nextMode) => setFileViewMode(nextMode),
  applyFileMode: () => applyFileMode(),
  openFilePickerSearchQuery: (query, options) => openFilePickerSearchQuery(query, options),
  setPreserveSearchOnFocus: (value) => filePickerMenuState.setPreserveSearchOnFocus(value),
  focusFilePickerInput: () => {
    try {
      filePickerInput.focus({ preventScroll: true });
    } catch (_) {
      filePickerInput.focus();
    }
  },
}));
const fileVideoPreviewRuntime = codoxearFileViewer.createFileVideoPreviewRuntime(wiring.createFileVideoPreviewOptions({
  controller: fileViewerController,
  fetchPreview: (url, options) => fetch(url, options),
  resolveAppUrl: (url) => resolveAppUrl(url),
  handleAuthLoss: () => handleAppAuthLoss(),
  errorText: (error) => codoxearFileHelpers.fileVideoPreviewErrorText(error),
  video: fileVideo,
}));
const fileLoadResultRuntime = codoxearFileViewer.createFileLoadResultRuntime(wiring.createFileLoadResultOptions({
  controller: fileViewerController,
  resolveAppUrl,
  setStatus: (status) => {
    fileStatus.textContent = status;
  },
  disposeFileEditor: () => disposeFileEditor(),
  renderMonacoDiff: (rel, originalText, modifiedText, lineNumber, request, options) => renderMonacoDiff(rel, originalText, modifiedText, lineNumber, request, options),
  renderMonacoFile: (rel, text, lineNumber, langOverride, request) => renderMonacoFile(rel, text, lineNumber, langOverride, request),
  renderMarkdownPreview: (rel, text) => renderMarkdownPreview(rel, text),
  renderBlockedFileNotice: (rel, reason, viewerMaxBytes, size) => renderBlockedFileNotice(rel, reason, viewerMaxBytes, size),
  renderPdfFile: (rel, url, request) => renderPdfFile(rel, url, request),
  showImage: (src, alt) => fileRenderSurfaceRuntime.showImage(src, alt),
  showVideo: (loadPlan, options) => fileRenderSurfaceRuntime.showVideo(loadPlan, options),
  loadCompatibleVideoPreview: (token, options) => fileVideoPreviewRuntime.loadCompatibleVideoPreview(token, options),
}));
const fileCandidateRefreshRuntime = codoxearFileViewer.createFileCandidateRefreshRuntime(wiring.createFileCandidateRefreshOptions({
  controller: fileViewerController,
  currentSessionId: () => currentFileViewerSessionId(),
  selectedSessionId: () => getSelected(),
  blockUnavailableFileAction: () => blockUnavailableFileAction(),
  isSessionCurrent: (sessionId, syncToken) => fileViewerLifecycleRuntime.isSessionCurrent(sessionId, syncToken),
  ttlMs: FILE_CANDIDATE_CACHE_TTL_MS,
  nowMs: () => Date.now(),
  collectMessageFileRefs: () => collectMessageFileRefs(),
  sessionFiles: (sessionId) => {
    const s = sessionId ? getSessionIndex().get(sessionId) : null;
    return listFromFilesField(s && s.files);
  },
  sessionFileRecords: (sessionId) => {
    const s = sessionId ? getSessionIndex().get(sessionId) : null;
    return listFromFileRecords(s && s.files);
  },
  sessionRelativePath: (rawPath, sessionId) => sessionRelativePath(rawPath, sessionId),
  api: (url) => api(url),
  normalizeFileApiPath: (value) => normalizeFileApiPath(value),
  renderMenu: () => renderFilePickerMenu(),
}));
const openedFileRuntime = codoxearFileViewer.createOpenedFileRuntime(wiring.createOpenedFileOptions({
  currentSessionId: () => currentFileViewerSessionId(),
  selectedSessionId: () => getSelected(),
  sessionRelativePath: (rawPath, sessionId) => sessionRelativePath(rawPath, sessionId),
  activeIdentity: () => currentActiveFileIdentity(),
  fileEntryForPath: (rel, gitPath, apiPath) => fileEntryForPath(rel, gitPath, apiPath),
  upsertFileEntry: (entry) => upsertFileEntry(entry),
  sessionById: (sessionId) => getSessionIndex().get(sessionId) || null,
  listFromFilesField: (files) => listFromFilesField(files),
  listFromFileRecords: (files) => listFromFileRecords(files),
  deleteCandidateCache: (sessionId) => fileViewerController.deleteFileCandidateCache(sessionId),
}));
const fileReferenceRuntime = codoxearFileViewer.createFileReferenceRuntime(wiring.createFileReferenceOptions({
  selectedSessionId: () => getSelected(),
  sessionById: (sessionId) => getSessionIndex().get(sessionId) || null,
  sessions: () => Array.from(getSessionIndex().values()),
  chatRoot: chatInner,
  ElementCtor: Element,
  sessionRelativePath: (rawPath, sessionId) => sessionRelativePath(rawPath, sessionId),
  listFromFilesField: (files) => listFromFilesField(files),
  listFromFileRecords: (files) => listFromFileRecords(files),
  normalizeFileApiPath: (value) => normalizeFileApiPath(value),
  normalizeLineNumber: (value) => normalizeLineNumber(value),
  parseLocalFileRef,
  showFileViewer: (options) => showFileViewer(options),
  selectSession: (sessionId) => getSessionLifecycleController().selectSession(sessionId),
  openDirectorySession: (options) => newSessionDialogController.open(options),
  setToast: (message) => setToast(message),
  api: (url, options) => api(url, options),
  el,
}));

async function openDraftFilePathWithGuard(path) {
  return await fileViewerController.openDraftFilePathWithGuard(path);
}

async function requestHideFileViewer() {
  return await fileViewerController.requestHideFileViewer();
}

async function handleFileDiffModeButtonPress() {
  return await fileViewerController.handleFileDiffModeButtonPress();
}

async function handleFilePreviewModeButtonPress() {
  return await fileViewerController.handleFilePreviewModeButtonPress();
}

async function handleFileEditButtonPress() {
  return await fileViewerController.handleFileEditButtonPress();
}

function activeFileDownloadApiPath() {
  return fileViewerController.activeFileDownloadApiPath();
}

function setFileViewMode(mode) {
  return fileViewerController.setFileViewMode(mode);
}

function applyFileMode() {
  return fileModeControlsRuntime.apply(fileViewerController.currentFileModeControlState());
}

function applyFileMenuState() {
  return filePickerDomRuntime.apply();
}

function resetFilePickerInput() {
  return filePickerDomRuntime.resetInput(activeFilePathValue() || "");
}

function closeFilePickerMenu({ restoreInput = false } = {}) {
  return filePickerDomRuntime.close({ restoreInput, inputValue: activeFilePathValue() || "" });
}

function filePickerSelectionLine() {
  return filePickerMenuState.selectionLine(filePickerInput.value);
}

function openFilePickerSearchQuery(query, { line = null, suppressDraft = false } = {}) {
  return filePickerInputRuntime.openSearchQuery(query, { line, suppressDraft });
}

function normalizeFileApiPath(value) {
  return typeof value === "string" && value !== "" ? value : "";
}

function setFilePath(rel, { line = null, gitPath = undefined, apiPath = undefined } = {}) {
  return fileViewerPanelRuntime.setFilePath(rel, { line, gitPath, apiPath });
}

function fileCandidateKey(path, gitPath = false, apiPath = "") {
  return fileViewerController.fileCandidateKey(path, gitPath, apiPath);
}

function fileEntryForPath(path, gitPath = false, apiPath = "") {
  return fileViewerController.fileEntryForPath(path, gitPath, apiPath);
}

async function openFilePathWithResolvedMode(path, { line = null, changed = null, isCurrent = null, gitPath = null, apiPath = "" } = {}) {
  return await fileViewerController.openFilePathWithResolvedMode(path, { line, changed, isCurrent, gitPath, apiPath });
}

function upsertFileEntry(entry) {
  return fileViewerController.upsertFileEntry(entry);
}

function rememberOpenedFile(relPath, absPath = null) {
  return openedFileRuntime.remember(relPath, absPath);
}

function collectMessageFileRefs() {
  return fileReferenceRuntime.collectMessageFileRefs();
}

function resetFileSearchState() {
  filePickerSearchState.reset();
}

function filePickerSearchSnapshot() {
  return filePickerSearchState.snapshot();
}

function renderFilePickerMenu() {
  return filePickerRenderRuntime.render();
}

async function upgradeCandidateFileRefs(root) {
  return await fileReferenceRuntime.upgradeCandidateRefs(root);
}

function sessionRelativePath(rawPath, sidOverride = null) {
  const sid = typeof sidOverride === "string" && sidOverride ? sidOverride : getSelected();
  const s = sid ? getSessionIndex().get(sid) : null;
  if (!s || !s.cwd) return null;
  const abs = stripPathLocationSuffix(rawPath);
  const cwd = String(s.cwd || "").replace(/\/+$/, "");
  if (!abs) return null;
  if (abs === cwd) return ".";
  if (abs.startsWith(cwd + "/")) return abs.slice(cwd.length + 1);
  return null;
}

async function refreshFileCandidates({ force = false, sessionId = null, syncToken = null } = {}) {
  return await fileCandidateRefreshRuntime.refresh({ force, sessionId, syncToken });
}

async function showFileViewer({ path = "", mode = "", manual = false, line = null, pickerQuery = "" } = {}) {
  void manual;
  if (selectedSessionLaunchFailed()) {
    setToast("failed launch has no file browser");
    return false;
  }
  return await fileViewerLifecycleRuntime.show({ path, mode, line, pickerQuery });
}
function hideFileViewer() {
  return fileViewerLifecycleRuntime.hide();
}
function handleFileViewerSessionUnavailable(sessionId) {
  return fileViewerController.handleFileViewerSessionUnavailable(sessionId);
}
async function applyFileLoadResult(rel, result, request, { viewMode = "file" } = {}) {
  return await fileLoadResultRuntime.apply(rel, result, request, { viewMode });
}

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
const fileTouchController = codoxearFileTouch.createFileTouchController(wiring.createFileTouchOptions({
  fileViewerController: () => fileViewerController,
}));
addAppEvent(document, "keydown", (event) => fileTouchController.handleFileTouchSelectionKeydown(event), true);
addAppEvent(document, "keydown", handleFileEditorSaveShortcut, true);
addAppEvent(document, "keydown", handleFileEditorDeleteKeydown, true);
addAppEvent(
  document,
  "beforeinput",
  (e) => {
    suppressFileEditorNativeDelete(e);
  },
  true
);
addAppEvent(
  document,
  "input",
  (e) => {
    suppressFileEditorNativeDelete(e);
  },
  true
);
addAppEvent(document, "keydown", (e) => {
  if (e.key === "Tab" && appConfirm.style.display === "flex") {
    const focusable = appConfirmFocusableControls();
    e.preventDefault();
    e.stopPropagation();
    if (!focusable.length) return;
    const currentIndex = focusable.indexOf(document.activeElement);
    const offset = e.shiftKey ? -1 : 1;
    const nextIndex = currentIndex < 0 ? (e.shiftKey ? focusable.length - 1 : 0) : (currentIndex + offset + focusable.length) % focusable.length;
    try {
      focusable[nextIndex].focus({ preventScroll: true });
    } catch {}
    return;
  }
  if (e.key !== "Escape") return;
  if (appConfirm.style.display === "flex") {
    e.preventDefault();
    e.stopPropagation();
    resolveAppConfirm(false);
    return;
  }
  if (filePasteDialogRuntime.isOpen()) {
    hideFilePasteDialog({ restoreFocus: true });
    return;
  }
  if (fileUnsavedDialog.style.display === "flex") {
    fileUnsavedController.hideFileUnsavedDialog("cancel");
    return;
  }
  if (isFileViewerOpen()) {
    e.preventDefault();
    void requestHideFileViewer();
    return;
  }
  if (sendChoice.style.display === "flex") {
    e.preventDefault();
    e.stopPropagation();
    closeSendChoiceDialog({ restoreFocus: true });
    return;
  }
  if (queueViewer.style.display === "flex") hideQueueViewer();
  if (helpViewer.style.display === "flex") hideHelpViewer();
  if (diagViewer.style.display === "flex") hideDiagViewer();
  if (voiceController.isSettingsOpen()) hideVoiceSettingsDialog();
  if (sessionEditController.viewer.style.display === "flex" || sessionEditController.viewer.open) sessionEditController.hideEditSession();
  if (newSessionDialogController.isOpen()) newSessionDialogController.close();
});


    return Object.freeze({
      dialogMenusController,
      filePickerSearchState,
      fileViewerController,
      fileUnsavedController,
      fileReferenceRuntime,
      hideFilePasteDialog,
      currentFileViewerSessionId,
      ensureCurrentFileViewerSession,
      currentFileDirty,
      isFileViewerOpen,
      handleFileViewerSessionUnavailable,
      refreshFileCandidates,
    });
  }

  global.CodoxearFileOps = Object.freeze({ createFileOpsController });
})(window);
