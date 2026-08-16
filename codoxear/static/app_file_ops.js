import * as CodoxearFileEditorOps from "./app_file_editor_ops.js";
import * as CodoxearFilePickerOps from "./app_file_picker_ops.js";
import * as CodoxearFileUnsaved from "./app_file_unsaved.js";
import * as CodoxearSessionEdit from "./app_session_edit.js";


/* File operations composition: viewer, editor, picker, unsaved-change, and touch keyboard. */

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file operations dependency missing: ${name}`);
    return value;
  }

  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`file operations dependency missing: ${name}`);
    return value;
  }

  async function copyToClipboard(text) {
    const nav = typeof navigator !== "undefined" ? navigator : window.navigator;
    if (!window.isSecureContext || !nav || !nav.clipboard || typeof nav.clipboard.writeText !== "function") {
      throw new Error("Clipboard API unavailable; requires a secure context (HTTPS)");
    }
    await nav.clipboard.writeText(String(text ?? ""));
  }

  function createFileEditModeController(options = {}) {
    const fileViewerController = typeof options.fileViewerController === "function" ? options.fileViewerController : null;
    if (!fileViewerController) throw new TypeError("file edit mode controller dependency missing: fileViewerController");

    function setFileEditMode(nextMode) {
      const viewer = fileViewerController();
      if (!viewer || typeof viewer.setFileEditMode !== "function") {
        throw new TypeError("file edit mode controller dependency missing: fileViewerController.setFileEditMode");
      }
      return viewer.setFileEditMode(nextMode);
    }

    return Object.freeze({ setFileEditMode });
  }

  function createFileTouchController(options = {}) {
    if (typeof options.fileViewerController !== "function") {
      throw new TypeError("file touch controller dependency missing: fileViewerController");
    }

    function handleFileTouchSelectionKeydown(event) {
      const controller = options.fileViewerController();
      if (!controller || typeof controller.handleFileTouchSelectionKeydown !== "function") {
        throw new TypeError("file touch controller dependency missing: fileViewerController.handleFileTouchSelectionKeydown");
      }
      return controller.handleFileTouchSelectionKeydown(event);
    }

    return Object.freeze({ handleFileTouchSelectionKeydown });
  }

  function createFileOpsController(options = {}) {
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function") throw new TypeError("file operations dependency missing: sessionState");
    const getSessionIndex = requireFunction(options.getSessionIndex, "getSessionIndex");
    const getSessionLifecycleController = requireFunction(options.getSessionLifecycleController, "getSessionLifecycleController");
    const {
      wiring, document, window, HTMLElement, requestAnimationFrame, setTimeout,
      $, el, iconSvg, resolveAppUrl, api, setToast, confirmApp, addAppEvent,
      normalizeLineNumber, markdownPreviewHtml,
      blockedFileMessage, listFromFilesField, listFromFileRecords, baseName,
      codoxearFilePicker, codoxearFilePickerOps, codoxearFileViewer, codoxearFileEditor, codoxearFileEditorOps, codoxearFileEditMode,
      codoxearFileTouch, codoxearDialogMenus,
      prepareModalOpen, afterModalVisibilityChanged, focusModalCloseButton, restoreModalFocus,
      isModalTargetOpen, newSessionDialogController, eventBindings,
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
    const filePickerOpsModule = codoxearFilePickerOps || CodoxearFilePickerOps;
    if (!filePickerOpsModule || typeof filePickerOpsModule.createFilePickerOpsController !== "function")
      throw new Error("Codoxear file picker operations controller failed to load");
    const filePickerController = filePickerOpsModule.createFilePickerOpsController(wiring.createFilePickerOpsOptions({
      wiring: wiring,
      codoxearFilePicker: codoxearFilePicker,
      normalizeLineNumber: normalizeLineNumber,
      filePickerField: filePickerField,
      filePickerMenu: filePickerMenu,
      filePickerInput: filePickerInput,
      api: api,
      document: document,
      el: el,
      sessionState,
      blockUnavailableFileAction: blockUnavailableFileAction,
      currentFileViewerSessionId: currentFileViewerSessionId,
      fileViewerController: () => fileViewerController,
      fileCandidateKey: (...args) => fileCandidateKey(...args),
      currentActiveFileDraft: currentActiveFileDraft,
      activeFilePathValue: activeFilePathValue,
      normalizeFileApiPath: (value) => normalizeFileApiPath(value),
      renderFilePickerMenu: () => renderFilePickerMenu(),
      applyFileMenuState: () => applyFileMenuState(),
      normalizeDraftFilePath: normalizeDraftFilePath,
      filePickerSectionLabel: filePickerSectionLabel,
      duplicateFilePickerPaths: duplicateFilePickerPaths,
      rawByteDuplicatePaths: rawByteDuplicatePaths,
      filePickerIdentityHint: filePickerIdentityHint,
      filePickerTitle: filePickerTitle,
      currentActiveFileIdentity: currentActiveFileIdentity,
      openDraftFilePathWithGuard: (...args) => openDraftFilePathWithGuard(...args),
      openFilePathWithResolvedMode: (...args) => openFilePathWithResolvedMode(...args),
      filePickerSelectionLine: () => filePickerSelectionLine(),
      ensureCurrentFileViewerSession: ensureCurrentFileViewerSession,
      resetFilePickerInput: () => resetFilePickerInput(),
      closeFilePickerMenu: (...args) => closeFilePickerMenu(...args),
      resetFileSearchState: () => resetFileSearchState(),
      setFileStatus: (status) => { fileStatus.textContent = status; },
      requestAnimationFrame: (callback) => requestAnimationFrame(callback),
    }));
    const { menuState: filePickerMenuState, domRuntime: filePickerDomRuntime, searchState: filePickerSearchState,
      entryRuntime: filePickerEntryRuntime, renderRuntime: filePickerRenderRuntime, inputRuntime: filePickerInputRuntime } = filePickerController;
    const fileEditorOpsModule = codoxearFileEditorOps || CodoxearFileEditorOps;
    if (!fileEditorOpsModule || typeof fileEditorOpsModule.createFileEditorOpsController !== "function")
      throw new Error("Codoxear file editor operations controller failed to load");
    const fileEditorOpsController = fileEditorOpsModule.createFileEditorOpsController(wiring.createFileEditorOpsOptions({
      addAppEvent: addAppEvent,
      wiring: wiring,
      codoxearFileEditor: codoxearFileEditor,
      resolveAppUrl: resolveAppUrl,
      fileDiff: fileDiff,
      normalizeLineNumber: normalizeLineNumber,
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
    const { runtime: fileEditorRuntime, monacoLoader: fileEditorMonacoLoader, renderer: fileEditorRenderer } = fileEditorOpsController;
const PDFJS_LOADER_TIMEOUT_MS = 6000;
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
  currentSessionId: () => currentFileViewerSessionId() || sessionState.get("selected") || "",
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
const fileUnsavedController = CodoxearFileUnsaved.createFileUnsavedController(wiring.createFileUnsavedOptions({
  documentTarget: document,
  ElementCtor: HTMLElement,
  dialogRuntime: fileUnsavedDialogRuntime,
  getFileViewerController: () => fileViewerController,
}));

function currentFileViewerSessionId() {
  return fileViewerController.currentFileViewerSessionId();
}

function currentFileSessionId() {
  return String(currentFileViewerSessionId() || sessionState.get("selected") || "").trim();
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
  sessionState,
  normalizeFileApiPath: (value) => normalizeFileApiPath(value),
  api: (url, options) => api(url, options),
}));

const fileViewerController = codoxearFileViewer.createFileViewerController(wiring.createFileViewerOptions({
  wiring,
  el,
  fileStatus,
  fileEditButton: fileEditBtn,
  iconSvg,
  currentSessionId: () => currentFileViewerSessionId(),
  currentFileSessionId: () => currentFileSessionId(),
  normalizeLineNumber,
  normalizeFileApiPath: (value) => normalizeFileApiPath(value),
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
  fileEditorDeleteCommandForKey: (key) => codoxearFileHelpers.fileEditorDeleteCommandForKey(key),
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
const sessionEditController = CodoxearSessionEdit.createSessionEditController(wiring.createSessionEditOptions({
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
  sessionState,
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
  sessionState,
  maybeHandleUnsavedFileChanges: () => fileUnsavedController.maybeHandleUnsavedFileChanges(),
  filePickerSearchSessionId: () => filePickerSearchSnapshot().sessionId,
  refreshFileCandidates: (options) => refreshFileCandidates(options),
  setFilePath: (path, options) => setFilePath(path, options),
  openFilePathWithResolvedMode: (path, options) => openFilePathWithResolvedMode(path, options),
  renderEmptyFileViewerTarget: (options) => renderEmptyFileViewerTarget(options),
  setStatus: (status) => {
    fileStatus.textContent = status;
  },
  showModal: ({ wasOpen = false, queryOpen = false } = {}) => fileViewerModalRuntime.show({
    wasOpen,
    queryOpen,
    activeElement: document.activeElement,
    ElementCtor: HTMLElement,
  }),
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
  sessionState,
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
  sessionState,
  sessionRelativePath: (rawPath, sessionId) => sessionRelativePath(rawPath, sessionId),
  activeIdentity: () => currentActiveFileIdentity(),
  fileEntryForPath: (rel, gitPath, apiPath) => fileViewerController.fileEntryForPath(rel, gitPath, apiPath),
  upsertFileEntry: (entry) => upsertFileEntry(entry),
  sessionById: (sessionId) => getSessionIndex().get(sessionId) || null,
  listFromFilesField: (files) => listFromFilesField(files),
  listFromFileRecords: (files) => listFromFileRecords(files),
  deleteCandidateCache: (sessionId) => fileViewerController.deleteFileCandidateCache(sessionId),
}));
const fileReferenceRuntime = codoxearFileViewer.createFileReferenceRuntime(wiring.createFileReferenceOptions({
  sessionState,
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

const filePickerOperations = filePickerOpsModule.createFilePickerOperationDelegates(wiring.createFilePickerOperationDelegatesOptions({
  fileViewerController: fileViewerController,
  fileModeControlsRuntime: fileModeControlsRuntime,
  filePickerDomRuntime: filePickerDomRuntime,
  filePickerMenuState: filePickerMenuState,
  filePickerInput: filePickerInput,
  filePickerInputRuntime: filePickerInputRuntime,
  activeFilePathValue: activeFilePathValue,
  openedFileRuntime: openedFileRuntime,
  fileReferenceRuntime: fileReferenceRuntime,
  filePickerSearchState: filePickerSearchState,
  filePickerRenderRuntime: filePickerRenderRuntime,
  fileViewerPanelRuntime: fileViewerPanelRuntime,
  sessionState,
  getSessionIndex: getSessionIndex,
  stripPathLocationSuffix: stripPathLocationSuffix,
}));
const { openDraftFilePathWithGuard, requestHideFileViewer, handleFileDiffModeButtonPress,
  handleFilePreviewModeButtonPress, handleFileEditButtonPress, activeFileDownloadApiPath, setFileViewMode,
  applyFileMode, applyFileMenuState, resetFilePickerInput, closeFilePickerMenu, filePickerSelectionLine,
  openFilePickerSearchQuery, normalizeFileApiPath, setFilePath, fileCandidateKey, fileEntryForPath,
  openFilePathWithResolvedMode, upsertFileEntry, rememberOpenedFile, collectMessageFileRefs,
  resetFileSearchState, filePickerSearchSnapshot, renderFilePickerMenu, upgradeCandidateFileRefs,
  sessionRelativePath } = filePickerOperations;

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

const fileTouchController = codoxearFileTouch.createFileTouchController(wiring.createFileTouchOptions({
  fileViewerController: () => fileViewerController,
}));
const filePickerOpsBinding = filePickerOpsModule;
if (!filePickerOpsBinding || typeof filePickerOpsBinding.bindFilePickerInteractions !== "function")
  throw new Error("Codoxear file picker bindings failed to load");
filePickerOpsBinding.bindFilePickerInteractions(wiring.createFilePickerInteractionsOptions({
  eventBindings: eventBindings,
  fileBtn: fileBtn,
  showFileViewer: showFileViewer,
  filePickerInput: filePickerInput,
  filePickerInputRuntime: filePickerInputRuntime,
  fileModeDiffBtn: fileModeDiffBtn,
  fileModePreviewBtn: fileModePreviewBtn,
  fileEditBtn: fileEditBtn,
  handleFileDiffModeButtonPress: handleFileDiffModeButtonPress,
  handleFilePreviewModeButtonPress: handleFilePreviewModeButtonPress,
  handleFileEditButtonPress: handleFileEditButtonPress,
  fileVideoPreviewBtn: fileVideoPreviewBtn,
  fileVideoPreviewRuntime: fileVideoPreviewRuntime,
  fileDownloadBtn: fileDownloadBtn,
  fileDownloadRuntime: fileDownloadRuntime,
  activeFileDownloadApiPath: activeFileDownloadApiPath,
  codoxearFileViewer: codoxearFileViewer,
  fileTouchSelectBtn: fileTouchSelectBtn,
  fileTouchCopyBtn: fileTouchCopyBtn,
  fileTouchPasteBtn: fileTouchPasteBtn,
  fileTouchUpBtn: fileTouchUpBtn,
  fileTouchLeftBtn: fileTouchLeftBtn,
  fileTouchDownBtn: fileTouchDownBtn,
  fileTouchRightBtn: fileTouchRightBtn,
  toggleFileTouchSelectionMode: toggleFileTouchSelectionMode,
  copyActiveFileSelection: copyActiveFileSelection,
  pasteFromClipboardIntoActiveFile: pasteFromClipboardIntoActiveFile,
  handleFileTouchMoveButtonPress: handleFileTouchMoveButtonPress,
  fileCloseBtn: fileCloseBtn,
  fileBackdrop: fileBackdrop,
  requestHideFileViewer: requestHideFileViewer,
  $: $,
  fileUnsavedController: fileUnsavedController,
  fileUnsavedBackdrop: fileUnsavedBackdrop,
  filePasteInput: filePasteInput,
  handleFilePasteInsert: handleFilePasteInsert,
  hideFilePasteDialog: hideFilePasteDialog,
  filePasteBackdrop: filePasteBackdrop,
  chatInner: chatInner,
  codeBlockCopyRuntime: codeBlockCopyRuntime,
  fileReferenceRuntime: fileReferenceRuntime,
  fileDiff: fileDiff,
  addAppEvent: addAppEvent,
  document: document,
  Element: Element,
  isFileViewerOpen: isFileViewerOpen,
  menuState: filePickerMenuState,
  closeFilePickerMenu: closeFilePickerMenu,
}));
fileEditorOpsController.bindInteractions({
  addAppEvent, document, appConfirm, appConfirmFocusableControls, resolveAppConfirm, filePasteDialogRuntime,
  hideFilePasteDialog, fileUnsavedDialog, fileUnsavedController, isFileViewerOpen, requestHideFileViewer,
  sendChoice, closeSendChoiceDialog, queueViewer, hideQueueViewer, helpViewer, hideHelpViewer, diagViewer,
  hideDiagViewer, voiceController, hideVoiceSettingsDialog, sessionEditController, newSessionDialogController,
  handleFileEditorSaveShortcut, handleFileEditorDeleteKeydown, suppressFileEditorNativeDelete, fileTouchController,
});


    return Object.freeze({
      dialogMenusController,
      sessionEditController,
      filePickerSearchState,
      fileViewerController,
      fileUnsavedController,
      fileReferenceRuntime,
      closeFilePickerMenu,
      hideFilePasteDialog,
      currentFileViewerSessionId,
      ensureCurrentFileViewerSession,
      currentFileDirty,
      isFileViewerOpen,
      handleFileViewerSessionUnavailable,
      refreshFileCandidates,
    });
  }

export { copyToClipboard, createFileEditModeController, createFileTouchController, createFileOpsController };
