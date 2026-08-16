  function requireFunction(value, name) { if (typeof value !== "function") throw new TypeError(`file viewer dependency missing: ${name}`); return value; }
  const BROWSER_SAFE_VIDEO_TYPES = new Set(["video/mp4", "video/webm", "video/ogg"]);
  const FILE_EDITOR_UNAVAILABLE_MESSAGE = "Editing is unavailable because the code editor failed to load. Read-only preview remains available.";
  function createFileViewerOperationsRuntime(options = {}) {
    const { el, fileStatus, fileEditButton, iconSvg, currentSessionId, currentFileSessionId, normalizeSessionId, normalizeFileApiPath, isFileViewerOpen, hideFileUnsavedDialog, resetFileSearchState, closeFilePickerMenu, isTextFileKind, isDiffableFileKind, confirmReload, promptUnsavedFileChoice, restoreFileEditorText, hideFileViewer, setFilePath, resetFileViewerPanel, applyFileLoadResult, normalizeDraftFilePath, inspectSessionFilePath, api, focusEditor, disposeOpenRender, isMarkdownPreviewable, updateFileTouchToolbar, useTouchFileEditorControls, hasActiveFileCodeEditor, hasBlockingFileEditorModal, isTextEntryTarget, eventTargetElement, normalizeFileEditorPosition, applyFileEditorSelection, isCollapsedFileSelection, positionAfterInsertedText, fileEditorEditSupportAvailable, updateFileDiffEditorOptions, showFilePasteDialog, hideFilePasteDialog, clipboardReadAvailable, readClipboardText, fileEditorDeleteCommandForKey, isActiveFileEditorInput, getActiveFileSelectionText, copyToClipboard, focusActiveFileCodeEditor, nowMs, setToast, renderMonacoFile, getFileEditorText, fmtBytes, applyFileMode, rememberOpenedFile, renderFilePickerMenu, currentFileViewMode, currentFileNonDiffMode, setFileViewMode, currentFileEditMode, currentFileEditorKind, setFileEditorKind, setFileEditMode, currentActiveFileKind, currentActiveFileText, currentActiveFileEditable, currentActiveFileVersion, currentActiveFileDraft, applyActiveFileTextState, applyActiveFileDiffState, applyActiveFileNonTextState, currentActiveFileIdentity, currentActiveFileLine, startFileOpenRequest, isCurrentFileOpenRequest, normalizeExplicitFileOpenMode, resolveFileOpenMode, isFileOpenAbortError, activeFileEntry, isGitFileCandidatePath, currentFileCandidateGitStateFresh, activeFileCanEnterEditMode, activeFileEditorWritable, activeFileEditorIdleTextWritable, currentFileEditorState, isUnavailable, blockUnavailableFileAction, fileEntryForPath, resetActiveFileBufferState, resolveFileOpenViewMode, activeFileEditorIdleWritable, isFileViewerSessionUnavailable, rememberActiveFileSelection, setActiveFileIdentity } = options;
    let activeSaveConflict = null, fileSaveSeq = 0, activeFileSaveToken = 0, fileSavePending = false, fileDirty = false, fileUnsavedPromptResolver = null, activeVideoFallback = null, activePdfRender = null, fileTouchSelectMode = false, fileTouchSelectAnchor = null, fileTouchSelectHead = null, fileTouchSelectGoalColumn = null, fileTouchDeleteNativeSuppressUntil = 0;
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

    function disposeActivePdfRender() {
      const state = takeActivePdfRenderState();
      if (!state) return false;
      if (state.observer) {
        try {
          state.observer.disconnect();
        } catch (_) {}
      }
      for (const task of state.renderTasks || []) {
        try {
          task.cancel();
        } catch (_) {}
      }
      if (state.loadingTask) {
        try {
          state.loadingTask.destroy();
        } catch (_) {}
      }
      return true;
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
      const kind = currentFileEditorKind();
      if (kind !== "file") return;
      const editor = focusEditor();
      if (!editor || typeof editor.updateOptions !== "function") return;
      editor.updateOptions({ readOnly: !activeFileEditorWritable() });
    }

    function activeFileEditorUnavailableReason() {
      const state = currentFileEditorState();
      if (!state.path || state.unavailable || state.savePending || state.editMode || state.viewMode !== "file") return "";
      if (!state.editable || !isTextFileKind(state.kind)) return "";
      return state.editorKind === "plain-fallback" ? FILE_EDITOR_UNAVAILABLE_MESSAGE : "";
    }

    function syncFileEditButtonDisabledReason(reason) {
      if (reason) {
        fileEditButton.setAttribute("aria-disabled", "true");
        return;
      }
      if (typeof fileEditButton.removeAttribute === "function") fileEditButton.removeAttribute("aria-disabled");
    }

    function updateFileEditButton() {
      const unavailable = isUnavailable();
      const savePending = isFileSavePending();
      const editUnavailableReason = activeFileEditorUnavailableReason();
      const canEdit = activeFileCanEnterEditMode();
      fileEditButton.disabled = unavailable || savePending || (!canEdit && !editUnavailableReason);
      const editMode = Boolean(currentFileEditMode());
      const dirty = Boolean(currentFileDirty());
      const saveStyle = editMode || savePending;
      // Compact 32px toolbar chrome: .active supplies the ink-on-paper inversion;
      // .primary would also impose the 44px full-size primary min-height.
      fileEditButton.classList.toggle("active", saveStyle);
      fileEditButton.classList.toggle("dirty", dirty);
      if (savePending) fileEditButton.innerHTML = iconSvg("save");
      else if (editMode) fileEditButton.innerHTML = iconSvg("save");
      else fileEditButton.innerHTML = iconSvg("edit");
      const label = unavailable
        ? "Session unavailable; copy edits before closing"
        : savePending
          ? "Saving file"
          : editMode
            ? "Save file"
            : editUnavailableReason || "Edit file";
      fileEditButton.title = label;
      fileEditButton.setAttribute("aria-label", label);
      syncFileEditButtonDisabledReason(editUnavailableReason);
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
      if (!save.draft && save.apiPath) body.path_token = save.apiPath;
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
      const kind = currentFileEditorKind();
      if (kind !== "file") {
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
      await openFilePath(identity.path, { line: currentActiveFileLine(), gitPath: identity.gitPath, apiPath: identity.apiPath });
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
      const editUnavailableReason = activeFileEditorUnavailableReason();
      if (editUnavailableReason) {
        fileStatus.textContent = editUnavailableReason;
        setToast(editUnavailableReason);
        updateFileEditButton();
        return false;
      }
      if (currentFileEditMode()) {
        await saveActiveFileEdits({ exitEditMode: true });
        return true;
      }
      if (currentFileViewMode() !== "file") {
        fileStatus.textContent = "Switch to File view before editing.";
        return false;
      }
      if (!currentActiveFileEditable() || !isTextFileKind(currentActiveFileKind())) return false;
      setFileEditMode(true);
      focusActiveFileCodeEditor();
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
      const tokenQuery = identity.apiPath ? `&path_token=${encodeURIComponent(identity.apiPath)}` : "";
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
      fileStatus.textContent = rendered.monacoUnavailable && rendered.status ? rendered.status : `${rel} - new file`;
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
        const versionsRes = await api(`/api/sessions/${request.sessionId}/git/file_versions?path=${encodeURIComponent(rel)}${pathTokenQuery}`, {
          signal: request.signal,
        });
        return Object.freeze({
          result: Object.freeze({
            kind: "diff",
            baseText: versionsRes && typeof versionsRes.base_text === "string" ? versionsRes.base_text : "",
            currentText: versionsRes && typeof versionsRes.current_text === "string" ? versionsRes.current_text : "",
            baseExists: versionsRes && versionsRes.base_exists,
            currentExists: versionsRes && versionsRes.current_exists,
          }),
          absPath: versionsRes && typeof versionsRes.abs_path === "string" ? versionsRes.abs_path : null,
        });
      }
      const gitPathQuery = request.gitPath ? "&git_path=1" : "";
      const pathTokenQuery = request.apiPath ? `&path_token=${encodeURIComponent(request.apiPath)}` : "";
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
      const ok = await confirmReload(`Reload ${savePath} from disk and discard your unsaved editor draft?`);
      if (!ok) return;
      fileStatus.textContent = `Reloading ${savePath}...`;
      const reloaded = await openFilePath(savePath, { line: currentActiveFileLine(), gitPath: currentActiveFileIdentity().gitPath, apiPath: currentActiveFileIdentity().apiPath });
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
      setActiveVideoFallback, clearActiveVideoFallback, currentActiveVideoFallback, setActivePdfRenderState, takeActivePdfRenderState, clearActivePdfRenderState, isActivePdfRenderState, disposeActivePdfRender, currentActiveVideoPreviewToken, prepareActiveVideoLoadResult, handleActiveVideoLoadError, handleActiveVideoLoadedMetadata, prepareFileLoadResult, beginCompatibleVideoPreview, completeCompatibleVideoPreview, failCompatibleVideoPreview, loadCompatibleVideoPreview, clearUsedCompatibleVideoPreview, currentFileModeControlState, syncFileEditorReadOnly, updateFileEditButton, isFileSavePending, currentFileDirty, setFileDirty, clearActiveFileSaveState, beginActiveFileSaveRequest, isCurrentActiveFileSaveRequest, markActiveFileSavePending, finishActiveFileSaveRequest, buildActiveFileSaveBody, renderActiveFileSaveError, applyActiveFileSaveSuccess, submitActiveFileSave, saveActiveFileEdits, prepareFileEditorTextRestore, finishFileEditorTextRestore, discardActiveFileEdits, isFileUnsavedPromptPending, fileUnsavedPromptPlan, beginFileUnsavedPrompt, resolveFileUnsavedPrompt, applyPlainTextFallbackState, maybeHandleUnsavedFileChanges, handleFileUnsavedSaveChoice, handleFileUnsavedDiscardChoice, handleFileUnsavedCancelChoice, setFileViewModeWithGuard, requestHideFileViewer, openFilePathWithGuard, openFilePathWithResolvedMode, openDraftFilePathWithGuard, openDraftFilePath, finalizeFileOpenSuccess, clearFileTouchSelectionState, currentFileTouchSelectMode, currentFileTouchToolbarState, resetFileTouchSelectionState, toggleFileTouchSelectionMode, handleFileTouchMoveButtonPress, moveFileTouchSelection, handleFileTouchSelectionKeydown, handleFileEditorDeleteKeydown, suppressFileEditorNativeDelete, insertIntoActiveFileEditor, pasteFromClipboardIntoActiveFile, handleFilePasteInsert, copyActiveFileSelection, handleFileDiffModeButtonPress, handleFilePreviewModeButtonPress, handleFileEditButtonPress, handleFileEditorSaveShortcut, handleFileVideoPreviewButtonPress, activeFileDownloadApiPath, openFilePath, applyDraftFileLoad, renderFileOpenError, renderDraftFileOpenError, fetchFileOpenResult, isSaveConflictCurrent, reloadSaveConflict, keepEditingSaveConflict, currentSaveConflict, renderSaveConflict
    });
  }

export { createFileViewerOperationsRuntime };
