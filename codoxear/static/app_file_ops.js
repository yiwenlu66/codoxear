import * as CodoxearFileEditorOps from "./app_file_editor_ops.js";
import * as CodoxearFileVim from "./app_file_vim.js";
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
    const {
      isFileViewerOpen, isTextFileKind, focusEditor, updateFileTouchToolbar,
      useTouchFileEditorControls, hasActiveFileCodeEditor, fileEditorShortcutBlocked,
      normalizeFileEditorPosition, applyFileEditorSelection, isCollapsedFileSelection,
      positionAfterInsertedText, fileEditorEditSupportAvailable, updateFileDiffEditorOptions,
      showFilePasteDialog, hideFilePasteDialog, clipboardReadAvailable, readClipboardText,
      fileEditorDeleteCommandForKey, isActiveFileEditorInput, getActiveFileSelectionText,
      copyToClipboard, focusActiveFileCodeEditor, nowMs, setToast, getFileEditorText,
      currentFileViewMode, currentActiveFileKind, currentActiveFileText,
      activeFileEditorWritable, activeFileEditorIdleWritable, activeFileEditorIdleTextWritable,
      blockUnavailableFileAction, eventTargetElement, syncFileEditorReadOnly, setFileDirty,
    } = options;
    let fileTouchSelectMode = false;
    let fileTouchSelectAnchor = null;
    let fileTouchSelectHead = null;
    let fileTouchSelectGoalColumn = null;
    let fileTouchDeleteNativeSuppressUntil = 0;

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

    return Object.freeze({
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
    });
  }

  function createFileOpsController(options = {}) {
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.subscribe !== "function") throw new TypeError("file operations dependency missing: sessionState");
    const sessionCatalog = options.sessionCatalog;
    if (!sessionCatalog || typeof sessionCatalog.get !== "function" || typeof sessionCatalog.subscribe !== "function") throw new TypeError("file operations dependency missing: sessionCatalog");
    const getSessionIndex = () => sessionCatalog.get("sessionIndex");
    const getSessionLifecycleController = requireFunction(options.getSessionLifecycleController, "getSessionLifecycleController");
    const {
      wiring, document, window, HTMLElement, requestAnimationFrame, setTimeout,
      $, el, iconSvg, resolveAppUrl, subscribeTheme, api, setToast, confirmApp, addAppEvent,
      normalizeLineNumber, markdownPreviewHtml,
      blockedFileMessage, listFromFilesField, listFromFileRecords, baseName,
      codoxearFilePicker, codoxearFilePickerOps, codoxearFileViewer, codoxearFileEditor, codoxearFileEditorOps, codoxearFileEditMode,
      codoxearFileVim, codoxearFileTouch, codoxearDialogMenus, hintModeController,
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
      sessionDisplayName, setPickerButtonContent, storageGetItem,
      storageSetItem, stripPathLocationSuffix, useTouchFileEditorControls,
      filePickerField, filePickerMenu, filePickerInput, fileStatus, fileDiff, fileImage,
      fileVideo, fileVideoPreviewBtn, fileTouchToolbar, fileTouchActions, fileTouchDpad,
      fileTouchCopyBtn, fileTouchPasteBtn, fileTouchSelectBtn, fileTouchUpBtn, fileTouchLeftBtn,
      fileTouchDownBtn, fileTouchRightBtn, fileModeDiffBtn, fileModePreviewBtn, fileDownloadBtn,
      fileBackdrop, fileViewer, fileCloseBtn, fileUnsavedBackdrop, fileUnsavedDialog,
      filePasteBackdrop, filePasteDialog, filePasteInput, fileEditBtn, fileVimModeChip, chatInner,
      codeBlockCopyRuntime, appConfirm, appConfirmFocusableControls, resolveAppConfirm,
      sendChoice, closeSendChoiceDialog, queueViewer, hideQueueViewer, helpViewer,
      hideHelpViewer, diagViewer, hideDiagViewer
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
      subscribeTheme: subscribeTheme,
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

function hasBlockingFileEditorModal() {
  return modalIsolationTargets.some((node) => node !== fileViewer && isModalTargetOpen(node));
}

function syncFileEditorReadOnly() {
  return fileViewerController.syncFileEditorReadOnly();
}

function updateFileTouchToolbar() {
  return fileTouchToolbarRuntime.update(fileTouchController.currentFileTouchToolbarState());
}

function clearFileTouchSelectionState() {
  return fileTouchController.clearFileTouchSelectionState();
}

function currentFileTouchSelectMode() {
  return fileTouchController.currentFileTouchSelectMode();
}

function resetFileTouchSelectionState(options) {
  return fileTouchController.resetFileTouchSelectionState(options);
}

function toggleFileTouchSelectionMode() {
  return fileTouchController.toggleFileTouchSelectionMode();
}

function handleFileTouchMoveButtonPress(direction) {
  return fileTouchController.handleFileTouchMoveButtonPress(direction);
}

function handleFileEditorSaveShortcut(e) {
  return fileViewerController.handleFileEditorSaveShortcut(e);
}

function handleFileEditorDeleteKeydown(e) {
  return fileTouchController.handleFileEditorDeleteKeydown(e);
}

function suppressFileEditorNativeDelete(e) {
  return fileTouchController.suppressFileEditorNativeDelete(e);
}

async function copyActiveFileSelection() {
  return await fileTouchController.copyActiveFileSelection();
}

function hideFilePasteDialog({ restoreFocus = false } = {}) {
  return filePasteDialogRuntime.hide({ restoreFocus });
}

function showFilePasteDialog() {
  return filePasteDialogRuntime.show();
}

async function pasteFromClipboardIntoActiveFile() {
  return await fileTouchController.pasteFromClipboardIntoActiveFile();
}

function handleFilePasteInsert(text) {
  return fileTouchController.handleFilePasteInsert(text);
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
  clearFileTouchSelectionState();
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
  hasBlockingFileEditorModal: () => hasBlockingFileEditorModal(),
  isTextEntryTarget: (target) => isTextEntryElement(target),
  eventTargetElement: (value) => value instanceof HTMLElement ? value : null,
  isActiveFileEditorInput: (target) => fileEditorRuntime.isActiveInput(currentFileEditorKind(), target, HTMLElement),
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
  vimNormalActive: () => Boolean(fileVimRef.controller && fileVimRef.controller.isNormalMode()),
  onFileEditModeChanged: () => {
    if (fileVimRef.controller) fileVimRef.controller.syncEditMode();
  },
}));
const fileTouchController = codoxearFileTouch.createFileTouchController(wiring.createFileTouchOptions({
  isFileViewerOpen: () => isFileViewerOpen(),
  isTextFileKind: (kind) => isTextFileKind(kind),
  focusEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind()),
  updateFileTouchToolbar: () => updateFileTouchToolbar(),
  useTouchFileEditorControls: () => useTouchFileEditorControls(),
  hasActiveFileCodeEditor: () => Boolean(fileEditorRuntime.activeCodeEditor(currentFileEditorKind())),
  fileEditorShortcutBlocked: (target) => fileViewerController.fileEditorShortcutBlocked(target),
  normalizeFileEditorPosition: (editor, position) => fileEditorRuntime.normalizePosition(editor, position),
  applyFileEditorSelection: (editor, cursor, anchor) => fileEditorRuntime.applySelection(editor, cursor, anchor, fileEditorMonacoLoader.selectionCtor()),
  isCollapsedFileSelection: (selection) => fileEditorRuntime.isCollapsedSelection(selection),
  positionAfterInsertedText: codoxearFileHelpers.positionAfterInsertedText,
  fileEditorEditSupportAvailable: () => fileEditorMonacoLoader.editSupportAvailable(),
  updateFileDiffEditorOptions: (options) => fileEditorRuntime.updateEditorOptions(currentFileEditorKind(), options),
  showFilePasteDialog: () => showFilePasteDialog(),
  hideFilePasteDialog: (options) => hideFilePasteDialog(options),
  clipboardReadAvailable: () => Boolean(window.isSecureContext && navigator.clipboard && typeof navigator.clipboard.readText === "function"),
  readClipboardText: () => navigator.clipboard.readText(),
  fileEditorDeleteCommandForKey: (key) => codoxearFileHelpers.fileEditorDeleteCommandForKey(key),
  isActiveFileEditorInput: (target) => fileEditorRuntime.isActiveInput(currentFileEditorKind(), target, HTMLElement),
  getActiveFileSelectionText: () => fileEditorRuntime.activeSelectionText(currentFileEditorKind()),
  copyToClipboard: (text) => copyToClipboard(text),
  focusActiveFileCodeEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind()),
  nowMs: () => Date.now(),
  setToast: (message) => setToast(message),
  getFileEditorText: () => getFileEditorText(),
  currentFileViewMode: () => fileViewerController.currentFileViewMode(),
  currentActiveFileKind: () => fileViewerController.currentActiveFileKind(),
  currentActiveFileText: () => fileViewerController.currentActiveFileText(),
  activeFileEditorWritable: () => fileViewerController.activeFileEditorWritable(),
  activeFileEditorIdleWritable: () => fileViewerController.activeFileEditorIdleWritable(),
  activeFileEditorIdleTextWritable: () => fileViewerController.activeFileEditorIdleTextWritable(),
  blockUnavailableFileAction: () => blockUnavailableFileAction(),
  eventTargetElement: (value) => value instanceof HTMLElement ? value : null,
  syncFileEditorReadOnly: () => syncFileEditorReadOnly(),
  setFileDirty: (dirty) => setFileDirty(dirty),
}));
// The vim controller's capture keydown listener is registered here, before
// fileEditorOpsController.bindInteractions() runs, so it sees keys ahead of
// every other document capture listener.
const fileVimRef = { controller: null };
const fileVimController = codoxearFileVim.createFileVimController(wiring.createFileVimOptions({
  addAppEvent: addAppEvent,
  document: document,
  fileVimModeChip: fileVimModeChip,
  fileDiff: fileDiff,
  isFileViewerOpen: () => isFileViewerOpen(),
  hasBlockingFileEditorModal: () => hasBlockingFileEditorModal(),
  hintModeActive: () => Boolean(hintModeController && hintModeController.isActive()),
  touchSelectActive: () => currentFileTouchSelectMode(),
  fileEditorShortcutBlocked: (target) => fileViewerController.fileEditorShortcutBlocked(target),
  currentFileEditMode: () => currentFileEditMode(),
  setFileEditMode: (mode) => fileEditModeController.setFileEditMode(mode),
  currentFileDirty: () => currentFileDirty(),
  currentFileEditorKind: () => currentFileEditorKind(),
  activeFileEditor: () => fileEditorRuntime.activeCodeEditor(currentFileEditorKind()),
  focusActiveFileEditor: () => fileEditorRuntime.focusActiveCodeEditor(currentFileEditorKind()),
  activeFileEditorInsertWritable: () => fileViewerController.activeFileEditorInsertWritable(),
  syncFileEditorReadOnly: () => syncFileEditorReadOnly(),
  getFileEditorText: () => getFileEditorText(),
  currentActiveFileText: () => currentActiveFileText(),
  setFileDirty: (dirty) => setFileDirty(dirty),
  setToast: (message) => setToast(message),
  enterHintMode: () => Boolean(hintModeController && hintModeController.enter()),
}));
fileVimRef.controller = fileVimController;
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
  finishHide: (state) => {
    fileViewerModalRuntime.finishHide(state);
    // Closing the viewer abandons any pending vim prefix (g/d) and sub-mode
    // so it cannot leak into the next viewer session.
    if (fileVimRef.controller) fileVimRef.controller.syncEditMode();
  },
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
  sessionCatalog,
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

function syncFileButtonState() {
  const selected = sessionState.get("selected");
  const blocked = Boolean(selected && selectedSessionLaunchFailed());
  const label = !selected ? "Select a session to view files" : blocked ? "Failed launch has no file browser" : "View file";
  fileBtn.disabled = !selected || blocked;
  fileBtn.title = label;
  fileBtn.setAttribute("aria-label", label);
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
  addAppEvent, document, appConfirm, appConfirmFocusableControls,
  handleFileEditorSaveShortcut, handleFileEditorDeleteKeydown, suppressFileEditorNativeDelete, fileTouchController,
});

const unsubscribeFileButtonSelected = sessionState.subscribe("selected", syncFileButtonState);
const unsubscribeFileButtonCatalog = sessionCatalog.subscribe("sessionIndex", syncFileButtonState);
syncFileButtonState();


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
      dispose() {
        unsubscribeFileButtonSelected();
        unsubscribeFileButtonCatalog();
      },
    });
  }

export { copyToClipboard, createFileEditModeController, createFileTouchController, createFileOpsController };
