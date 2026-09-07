
  function isModalTargetOpen(node) {
    if (!node) return false;
    if (typeof HTMLDialogElement !== "undefined" && node instanceof HTMLDialogElement && node.open) return true;
    return !!(node.style && node.style.display && node.style.display !== "none");
  }

  function syncModalIsolation(app, targets) {
    const active = Array.isArray(targets) && targets.some(isModalTargetOpen);
    app.toggleAttribute("inert", active);
    if (active) app.setAttribute("aria-hidden", "true");
    else app.removeAttribute("aria-hidden");
    return active;
  }

  function restoreModalFocus(target, isStillOpen, requestFrame = requestAnimationFrame) {
    if (!target || !target.isConnected || typeof target.focus !== "function") return;
    if (typeof target.disabled === "boolean" && target.disabled) return;
    requestFrame(() => {
      if (typeof isStillOpen === "function" && isStillOpen()) return;
      try {
        target.focus({ preventScroll: true });
      } catch {}
    });
  }

  // Modal-open focus contract: an opened modal focuses its surface, never a
  // control inside it. Focusing the close button painted the theme focus ring
  // on WebKit (script focus with no prior focus matches :focus-visible), so
  // every dialog opened with a phantom accent outline on touch. The surface
  // carries tabindex=-1, Tab still reaches the controls, and the focus-ring
  // CSS targets button/input/textarea/select only — a surface cannot ring.
  function focusModalSurface(viewer, requestFrame = requestAnimationFrame) {
    requestFrame(() => {
      if (!isModalTargetOpen(viewer)) return;
      try {
        viewer.focus({ preventScroll: true });
      } catch {}
    });
  }

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`modal dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object" || !value.style) throw new TypeError(`modal dependency missing: ${name}`);
    return value;
  }

  // Named owner for cross-dialog policy and the shared picker button DOM shape.
  // Composition supplies feature-specific close capabilities; this controller
  // decides which transient surfaces close together and when app isolation syncs.
  function createModalPolicyController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("modal dependency missing: options");
    const app = requireNode(options.app, "app");
    const modalTargets = requireFunction(options.modalTargets, "modalTargets");
    const closeUnattended = requireFunction(options.closeUnattended, "closeUnattended");
    const isUnattendedOpen = requireFunction(options.isUnattendedOpen, "isUnattendedOpen");
    const closeSearch = requireFunction(options.closeSearch, "closeSearch");
    const isSearchOpen = requireFunction(options.isSearchOpen, "isSearchOpen");
    const isSidebarOpen = requireFunction(options.isSidebarOpen, "isSidebarOpen");
    const closeSidebar = requireFunction(options.closeSidebar, "closeSidebar");
    const closeFilePicker = requireFunction(options.closeFilePicker, "closeFilePicker");
    const closeNewSessionMenus = requireFunction(options.closeNewSessionMenus, "closeNewSessionMenus");
    const closeSessionDependencyMenu = requireFunction(options.closeSessionDependencyMenu, "closeSessionDependencyMenu");
    const el = requireFunction(options.el, "el");
    const iconSvg = requireFunction(options.iconSvg, "iconSvg");

    function closeTransientOverlays({ closeSearch: shouldCloseSearch = false } = {}) {
      if (isUnattendedOpen()) closeUnattended();
      if (shouldCloseSearch && isSearchOpen()) closeSearch();
      if (isSidebarOpen()) closeSidebar();
      closeFilePicker();
      closeNewSessionMenus();
      closeSessionDependencyMenu();
    }

    function prepareModalOpen(options = {}) {
      closeTransientOverlays(options);
    }

    function afterModalVisibilityChanged() {
      return syncModalIsolation(app, modalTargets());
    }

    function setPickerButtonContent(button, primaryText, secondaryText = "", placeholder = false) {
      if (!button) return;
      button.innerHTML = "";
      const textWrap = el("span", { class: `pickerButtonText${placeholder ? " placeholder" : ""}` });
      textWrap.appendChild(el("span", { class: "pickerButtonPrimary", text: String(primaryText || "") }));
      if (secondaryText) textWrap.appendChild(el("span", { class: "pickerButtonSecondary", text: String(secondaryText) }));
      button.appendChild(textWrap);
      button.appendChild(el("span", { class: "pickerButtonChevron", html: iconSvg("chevronDown") }));
    }

    return Object.freeze({
      afterModalVisibilityChanged,
      closeTransientOverlays,
      focusModalSurface,
      isModalTargetOpen,
      prepareModalOpen,
      restoreModalFocus,
      setPickerButtonContent,
    });
  }

  // Confirmation belongs to the modal domain because it owns a dialog's
  // visibility, promise settlement, initial focus, return focus, and controls.
  function createConfirmationController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("confirmation dependency missing: options");
    const backdrop = requireNode(options.backdrop, "backdrop");
    const viewer = requireNode(options.viewer, "viewer");
    const title = requireNode(options.title, "title");
    const message = requireNode(options.message, "message");
    const confirmButton = requireNode(options.confirmButton, "confirmButton");
    const cancelButton = requireNode(options.cancelButton, "cancelButton");
    const documentTarget = options.documentTarget;
    const ElementCtor = options.ElementCtor;
    if (!documentTarget || typeof documentTarget !== "object") throw new TypeError("confirmation dependency missing: documentTarget");
    if (typeof ElementCtor !== "function") throw new TypeError("confirmation dependency missing: ElementCtor");
    const requestFrame = requireFunction(options.requestFrame, "requestFrame");
    const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
    const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
    const addEvent = requireFunction(options.addEvent, "addEvent");

    let pending = null;
    let returnFocusElement = null;

    function normalize(confirmOptions = {}) {
      if (typeof confirmOptions === "string") {
        return { title: "Confirm action", message: confirmOptions, confirmText: "Confirm", cancelText: "Cancel", destructive: false };
      }
      const raw = confirmOptions && typeof confirmOptions === "object" ? confirmOptions : {};
      return {
        title: String(raw.title || "Confirm action"),
        message: String(raw.message || ""),
        confirmText: String(raw.confirmText || "Confirm"),
        cancelText: String(raw.cancelText || "Cancel"),
        destructive: Boolean(raw.destructive),
      };
    }

    function focusableControls() {
      return [cancelButton, confirmButton].filter((control) => control && !control.disabled && typeof control.focus === "function");
    }

    function focusInitial({ destructive = false } = {}) {
      requestFrame(() => {
        if (viewer.style.display !== "flex") return;
        const preferred = destructive ? cancelButton : confirmButton;
        const fallback = destructive ? confirmButton : cancelButton;
        const target = preferred && !preferred.disabled ? preferred : fallback && !fallback.disabled ? fallback : null;
        if (!target || typeof target.focus !== "function") return;
        try {
          target.focus({ preventScroll: true });
        } catch {}
      });
    }

    function resolve(result, { restoreFocus = true } = {}) {
      const current = pending;
      const target = returnFocusElement;
      pending = null;
      returnFocusElement = null;
      backdrop.style.display = "none";
      viewer.style.display = "none";
      afterModalVisibilityChanged();
      if (restoreFocus) restoreModalFocus(target, () => viewer.style.display === "flex", requestFrame);
      if (current && !current.settled) {
        current.settled = true;
        current.resolve(Boolean(result));
      }
    }

    function confirm(confirmOptions = {}) {
      if (pending) resolve(false, { restoreFocus: false });
      const normalized = normalize(confirmOptions);
      prepareModalOpen();
      title.textContent = normalized.title;
      message.textContent = normalized.message;
      confirmButton.textContent = normalized.confirmText;
      cancelButton.textContent = normalized.cancelText;
      returnFocusElement = documentTarget.activeElement instanceof ElementCtor ? documentTarget.activeElement : null;
      backdrop.style.display = "block";
      viewer.style.display = "flex";
      afterModalVisibilityChanged();
      focusInitial(normalized);
      return new Promise((promiseResolve) => {
        pending = { resolve: promiseResolve, settled: false };
      });
    }

    addEvent(confirmButton, "click", () => resolve(true));
    addEvent(cancelButton, "click", () => resolve(false));
    addEvent(backdrop, "click", () => resolve(false));

    return Object.freeze({ confirm, focusableControls, isOpen: () => viewer.style.display === "flex", resolve });
  }

  function modalButtonLabel(button) {
    return [button.textContent, button.getAttribute("aria-label")]
      .map((label) => String(label || "").trim().toLowerCase())
      .find(Boolean) || "";
  }

  function modalButtonHint(label, labels) {
    for (let index = 0; index < label.length; index += 1) {
      const candidate = label[index];
      if (!/[a-z0-9]/.test(candidate)) continue;
      if (labels.filter((other) => other[index] === candidate).length === 1) return candidate;
    }
    return "";
  }

  function createModalKeyboardHandler({ modalIsolationTargets, isTextEntryElement: isTextEntry }) {
    return function activateModalButtonForKey(e) {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.isComposing) return false;
      const key = String(e.key || "").toLowerCase();
      if (key.length !== 1 || isTextEntry(e.target)) return false;
      for (const modal of modalIsolationTargets) {
        if (!isModalTargetOpen(modal)) continue;
        const buttons = [...modal.querySelectorAll("button")].filter((button) => !button.disabled && !button.hidden && button.getClientRects().length);
        const labels = buttons.map(modalButtonLabel);
        const button = buttons.find((candidate, index) => modalButtonHint(labels[index], labels) === key);
        if (!button) continue;
        e.preventDefault();
        e.stopPropagation();
        button.click();
        return true;
      }
      return false;
    };
  }

export {
  isModalTargetOpen,
  syncModalIsolation,
  restoreModalFocus,
  focusModalSurface,
  createModalPolicyController,
  createConfirmationController,
  createModalKeyboardHandler,
};
