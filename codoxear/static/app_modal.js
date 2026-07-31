(function () {
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

  window.CodoxearModal = Object.freeze({
    isModalTargetOpen,
    syncModalIsolation,
    restoreModalFocus,
    focusModalCloseButton,
  });
})();
