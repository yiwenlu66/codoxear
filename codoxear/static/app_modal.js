  "use strict";

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

  function focusModalCloseButton(viewer, closeBtn, requestFrame = requestAnimationFrame) {
    requestFrame(() => {
      if (!isModalTargetOpen(viewer)) return;
      try {
        closeBtn.focus({ preventScroll: true });
      } catch {}
    });
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

export { isModalTargetOpen, syncModalIsolation, restoreModalFocus, focusModalCloseButton, createModalKeyboardHandler };
