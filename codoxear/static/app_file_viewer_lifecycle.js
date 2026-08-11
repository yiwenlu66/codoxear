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

  function requireRenderHostNode(value, name) {
    if (!value || !("innerHTML" in value) || typeof value.appendChild !== "function") {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requirePasteInput(value) {
    if (!value || !("value" in value)) throw new TypeError("file viewer dependency missing: filePasteInput");
    return value;
  }

  function requireVideoNode(value) {
    if (!value || !value.style || typeof value.removeAttribute !== "function" || typeof value.load !== "function") {
      throw new TypeError("file viewer dependency missing: fileVideo");
    }
    return value;
  }

  function requireImageNode(value) {
    if (!value || !value.style || typeof value.removeAttribute !== "function") {
      throw new TypeError("file viewer dependency missing: fileImage");
    }
    return value;
  }

  function requireToggleClassNode(value, name) {
    if (!value || !value.classList || typeof value.classList.toggle !== "function") {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requireModeControlButton(value, name) {
    if (
      !value ||
      !value.classList ||
      typeof value.classList.toggle !== "function" ||
      !value.style ||
      typeof value.setAttribute !== "function" ||
      !("disabled" in value)
    ) {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requireModalHostNode(value, name) {
    if (!value || typeof value.setAttribute !== "function" || typeof value.removeAttribute !== "function") {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requireTextNode(value, name) {
    if (!value || !("textContent" in value)) throw new TypeError(`file viewer dependency missing: ${name}`);
    return value;
  }

  function requireUnsavedButtonNode(value, name) {
    if (!value || !("hidden" in value) || !("disabled" in value) || !("textContent" in value)) {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }


  function createFileViewerLifecycleRuntime(options = {}) {
    const controller = options.controller || null;
    if (!controller) throw new TypeError("file viewer dependency missing: controller");
    const invalidateSessionSync = requireFunction(controller.invalidateFileViewerSessionSync, "controller.invalidateFileViewerSessionSync").bind(controller);
    const cancelPendingFileOpen = requireFunction(controller.cancelPendingFileOpen, "controller.cancelPendingFileOpen").bind(controller);
    const rememberActiveFileSelection = requireFunction(controller.rememberActiveFileSelection, "controller.rememberActiveFileSelection").bind(controller);
    const clearFileViewerSessionId = requireFunction(controller.clearFileViewerSessionId, "controller.clearFileViewerSessionId").bind(controller);
    const clearFileViewerUnavailableSession = requireFunction(controller.clearFileViewerUnavailableSession, "controller.clearFileViewerUnavailableSession").bind(controller);
    const clearActiveFileIdentity = requireFunction(controller.clearActiveFileIdentity, "controller.clearActiveFileIdentity").bind(controller);
    const beginHide = requireFunction(options.beginHide, "beginHide");
    const hideDisplay = requireFunction(options.hideDisplay, "hideDisplay");
    const finishHide = requireFunction(options.finishHide, "finishHide");
    const hideFileUnsavedDialog = requireFunction(options.hideFileUnsavedDialog, "hideFileUnsavedDialog");
    const hideFilePasteDialog = requireFunction(options.hideFilePasteDialog, "hideFilePasteDialog");
    const resetFileViewerPanel = requireFunction(options.resetFileViewerPanel, "resetFileViewerPanel");
    const closeFilePickerMenu = requireFunction(options.closeFilePickerMenu, "closeFilePickerMenu");
    const resetFileSearchState = requireFunction(options.resetFileSearchState, "resetFileSearchState");
    const setFileSearchSessionId = requireFunction(options.setFileSearchSessionId, "setFileSearchSessionId");
    const updateFileTouchToolbar = requireFunction(options.updateFileTouchToolbar, "updateFileTouchToolbar");

    function hide() {
      const hideState = beginHide();
      invalidateSessionSync();
      cancelPendingFileOpen();
      hideFileUnsavedDialog();
      hideFilePasteDialog();
      rememberActiveFileSelection();
      resetFileViewerPanel();
      closeFilePickerMenu({ restoreInput: true });
      resetFileSearchState();
      setFileSearchSessionId("");
      hideDisplay();
      clearFileViewerSessionId();
      clearFileViewerUnavailableSession();
      clearActiveFileIdentity();
      updateFileTouchToolbar();
      finishHide(hideState);
      return true;
    }

    function ensureSessionDeps() {
      return {
        isFileViewerOpen: requireFunction(options.isFileViewerOpen, "isFileViewerOpen"),
        selectedSessionId: requireFunction(options.selectedSessionId, "selectedSessionId"),
        maybeHandleUnsavedFileChanges: requireFunction(options.maybeHandleUnsavedFileChanges, "maybeHandleUnsavedFileChanges"),
        filePickerSearchSessionId: requireFunction(options.filePickerSearchSessionId, "filePickerSearchSessionId"),
        refreshFileCandidates: requireFunction(options.refreshFileCandidates, "refreshFileCandidates"),
        setFilePath: requireFunction(options.setFilePath, "setFilePath"),
        openFilePathWithResolvedMode: requireFunction(options.openFilePathWithResolvedMode, "openFilePathWithResolvedMode"),
        renderEmptyFileViewerTarget: requireFunction(options.renderEmptyFileViewerTarget, "renderEmptyFileViewerTarget"),
        setStatus: requireFunction(options.setStatus, "setStatus"),
      };
    }

    function sessionTransitionDeps() {
      return {
        currentViewerSessionId: requireFunction(controller.currentFileViewerSessionId, "controller.currentFileViewerSessionId").bind(controller),
        isCurrentSync: requireFunction(controller.isCurrentFileViewerSessionSync, "controller.isCurrentFileViewerSessionSync").bind(controller),
        beginSessionSync: requireFunction(controller.beginFileViewerSessionSync, "controller.beginFileViewerSessionSync").bind(controller),
        setViewerSessionId: requireFunction(controller.setFileViewerSessionId, "controller.setFileViewerSessionId").bind(controller),
        clearUnavailable: requireFunction(controller.clearFileViewerUnavailableSession, "controller.clearFileViewerUnavailableSession").bind(controller),
        resolveOpenTarget: requireFunction(controller.resolveFileViewerOpenTarget, "controller.resolveFileViewerOpenTarget").bind(controller),
      };
    }

    function isSelectionCurrent(sessionId, token = null) {
      const deps = ensureSessionDeps();
      const transition = sessionTransitionDeps();
      const sid = String(sessionId || "").trim();
      return Boolean(
        sid &&
          deps.isFileViewerOpen() &&
          String(deps.selectedSessionId() || "").trim() === sid &&
          (token === null || transition.isCurrentSync(token))
      );
    }

    function isSessionCurrent(sessionId, token = null) {
      const transition = sessionTransitionDeps();
      const sid = String(sessionId || "").trim();
      return Boolean(isSelectionCurrent(sid, token) && transition.currentViewerSessionId() === sid);
    }

    async function ensureCurrentSession() {
      const deps = ensureSessionDeps();
      if (!deps.isFileViewerOpen()) return true;
      const sid = String(deps.selectedSessionId() || "").trim();
      if (!sid) return false;
      const transition = sessionTransitionDeps();
      if (transition.currentViewerSessionId() === sid) return true;
      const syncToken = transition.beginSessionSync();
      if (!(await deps.maybeHandleUnsavedFileChanges())) return false;
      if (!isSelectionCurrent(sid, syncToken)) return false;
      cancelPendingFileOpen();
      rememberActiveFileSelection(transition.currentViewerSessionId());
      transition.setViewerSessionId(sid);
      transition.clearUnavailable();
      if (deps.filePickerSearchSessionId() !== transition.currentViewerSessionId()) {
        resetFileSearchState();
        setFileSearchSessionId(transition.currentViewerSessionId());
      }
      await deps.refreshFileCandidates({ sessionId: sid, syncToken });
      if (!isSessionCurrent(sid, syncToken)) return false;
      const target = transition.resolveOpenTarget({ sessionId: sid });
      if (target.kind === "path") {
        deps.setFilePath(target.path, { line: target.line, gitPath: target.gitPath, apiPath: target.apiPath });
        try {
          await deps.openFilePathWithResolvedMode(target.path, { line: target.line, changed: target.changed, gitPath: target.gitPath, apiPath: target.apiPath, isCurrent: () => isSessionCurrent(sid, syncToken) });
        } catch (error) {
          if (!isSessionCurrent(sid, syncToken)) return false;
          deps.setStatus(`error: ${error && error.message ? error.message : "unable to inspect path"}`);
        }
        return isSessionCurrent(sid, syncToken);
      }
      if (!isSessionCurrent(sid, syncToken)) return false;
      deps.renderEmptyFileViewerTarget({ updateTouchToolbar: true });
      return true;
    }

    function showDeps() {
      return {
        showModal: requireFunction(options.showModal, "showModal"),
        updateFileTouchToolbar: requireFunction(options.updateFileTouchToolbar, "updateFileTouchToolbar"),
        setFileViewMode: requireFunction(options.setFileViewMode, "setFileViewMode"),
        applyFileMode: requireFunction(options.applyFileMode, "applyFileMode"),
        resetFileViewerPanel: requireFunction(options.resetFileViewerPanel, "resetFileViewerPanel"),
        openFilePickerSearchQuery: requireFunction(options.openFilePickerSearchQuery, "openFilePickerSearchQuery"),
        setPreserveSearchOnFocus: requireFunction(options.setPreserveSearchOnFocus, "setPreserveSearchOnFocus"),
        focusFilePickerInput: requireFunction(options.focusFilePickerInput, "focusFilePickerInput"),
      };
    }

    async function show({ path = "", mode = "", line = null, pickerQuery = "" } = {}) {
      const deps = ensureSessionDeps();
      const transition = sessionTransitionDeps();
      const ui = showDeps();
      const wasOpen = deps.isFileViewerOpen();
      if (wasOpen && !(await deps.maybeHandleUnsavedFileChanges())) return false;
      cancelPendingFileOpen();
      const explicitPath = String(path ?? "");
      const query = String(pickerQuery ?? "");
      const queryOpen = !explicitPath && query !== "";
      ui.showModal({ wasOpen, queryOpen });
      ui.updateFileTouchToolbar();
      rememberActiveFileSelection(transition.currentViewerSessionId());
      const sid = String(deps.selectedSessionId() || "").trim();
      const syncToken = transition.beginSessionSync();
      transition.setViewerSessionId(sid);
      transition.clearUnavailable();
      if (deps.filePickerSearchSessionId() !== transition.currentViewerSessionId()) {
        resetFileSearchState();
        setFileSearchSessionId(transition.currentViewerSessionId());
      }
      if (mode === "file" || mode === "diff" || mode === "preview") ui.setFileViewMode(mode);
      else ui.applyFileMode();
      if (queryOpen) {
        ui.resetFileViewerPanel();
        clearActiveFileIdentity({ line });
        deps.setStatus("Choose which file to open.");
        ui.openFilePickerSearchQuery(query, { line, suppressDraft: true });
        ui.setPreserveSearchOnFocus(true);
      }
      await deps.refreshFileCandidates({ sessionId: sid, syncToken });
      if (!isSessionCurrent(sid, syncToken)) return false;
      if (queryOpen) {
        ui.focusFilePickerInput();
        return true;
      }
      const target = transition.resolveOpenTarget({ sessionId: sid, explicitPath, explicitLine: line });
      if (target.kind === "path") {
        deps.setFilePath(target.path, { line: target.line, gitPath: target.gitPath, apiPath: target.apiPath });
        void deps.openFilePathWithResolvedMode(target.path, { line: target.line, changed: target.changed, gitPath: target.gitPath, apiPath: target.apiPath, isCurrent: () => isSessionCurrent(sid, syncToken) }).catch((error) => {
          if (!isSessionCurrent(sid, syncToken)) return;
          deps.setStatus(`error: ${error && error.message ? error.message : "unable to inspect path"}`);
        });
        return true;
      }
      deps.renderEmptyFileViewerTarget();
      return true;
    }

    return Object.freeze({ ensureCurrentSession, hide, isSelectionCurrent, isSessionCurrent, show });
  }

export { createFileViewerLifecycleRuntime };
