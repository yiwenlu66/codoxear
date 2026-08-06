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


  function createFileUnsavedDialogRuntime(options = {}) {
    const backdrop = requireStyledNode(options.backdrop, "fileUnsavedBackdrop");
    const dialog = requireStyledNode(options.dialog, "fileUnsavedDialog");
    const viewer = requireModalHostNode(options.viewer, "fileViewer");
    const title = requireTextNode(options.title, "fileUnsavedTitle");
    const message = requireTextNode(options.message, "fileUnsavedMessage");
    const saveButton = requireUnsavedButtonNode(options.saveButton, "fileUnsavedSaveButton");
    const discardButton = requireUnsavedButtonNode(options.discardButton, "fileUnsavedDiscardButton");
    const cancelButton = requireUnsavedButtonNode(options.cancelButton, "fileUnsavedCancelButton");
    const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
    const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
    const restoreModalFocus = requireFunction(options.restoreModalFocus, "restoreModalFocus");
    const isModalTargetOpen = requireFunction(options.isModalTargetOpen, "isModalTargetOpen");
    const requestFrame = requireFunction(options.requestAnimationFrame, "requestAnimationFrame");
    const promptPlan = requireFunction(options.promptPlan, "promptPlan");
    const beginPrompt = requireFunction(options.beginPrompt, "beginPrompt");
    const resolvePrompt = requireFunction(options.resolvePrompt, "resolvePrompt");
    const setReturnFocusElement = requireFunction(options.setReturnFocusElement, "setReturnFocusElement");
    const takeReturnFocusElement = requireFunction(options.takeReturnFocusElement, "takeReturnFocusElement");
    const isUnavailable = requireFunction(options.isUnavailable, "isUnavailable");

    function syncMode() {
      const unavailable = Boolean(isUnavailable());
      title.textContent = unavailable ? "Session unavailable" : "Unsaved changes";
      message.textContent = unavailable
        ? "This session is no longer available. Copy your edits before closing; they cannot be saved here."
        : "Save this file before leaving the editor?";
      saveButton.hidden = unavailable;
      saveButton.disabled = unavailable;
      discardButton.textContent = unavailable ? "Close without saving" : "Discard";
      return Object.freeze({ unavailable });
    }

    function focusInitialControl() {
      requestFrame(() => {
        if (!isModalTargetOpen(dialog)) return;
        const target = saveButton && !saveButton.hidden && !saveButton.disabled ? saveButton : discardButton || cancelButton;
        if (!target || typeof target.focus !== "function") return;
        try {
          target.focus({ preventScroll: true });
        } catch (_) {}
      });
      return true;
    }

    function hide(choice = "cancel") {
      const focusTarget = takeReturnFocusElement();
      backdrop.style.display = "none";
      dialog.style.display = "none";
      viewer.removeAttribute("inert");
      viewer.removeAttribute("aria-hidden");
      afterModalVisibilityChanged();
      restoreModalFocus(focusTarget, () => isModalTargetOpen(dialog) || !isModalTargetOpen(viewer));
      resolvePrompt(choice);
      return true;
    }

    function promptChoice(activeElement = null, ElementCtor = null) {
      const plan = promptPlan();
      if (plan.kind === "choice") return Promise.resolve(plan.choice);
      prepareModalOpen();
      setReturnFocusElement(activeElement, ElementCtor);
      syncMode();
      viewer.setAttribute("inert", "");
      viewer.setAttribute("aria-hidden", "true");
      backdrop.style.display = "block";
      dialog.style.display = "flex";
      afterModalVisibilityChanged();
      focusInitialControl();
      return beginPrompt();
    }

    return Object.freeze({ focusInitialControl, hide, promptChoice, syncMode });
  }


  window.CodoxearFileUnsavedDialog = Object.freeze({ createFileUnsavedDialogRuntime });
})();
