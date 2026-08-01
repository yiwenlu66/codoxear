(function () {
  "use strict";

  const SESSION_HINTS = Object.freeze(["1", "2", "3", "4", "5", "6", "7", "8", "9"]);

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`hint mode controller dependency missing: ${name}`);
    return value;
  }

  function requireArray(value, name) {
    if (!Array.isArray(value)) throw new TypeError(`hint mode controller dependency missing: ${name}`);
    return value;
  }

  function requireDocument(value) {
    if (!value || typeof value.querySelectorAll !== "function" || typeof value.createElement !== "function" || !value.body)
      throw new TypeError("hint mode controller dependency missing: documentTarget");
    return value;
  }

  function createHintModeController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("hint mode controller dependency missing: options");

    const documentTarget = requireDocument(options.documentTarget || (typeof document !== "undefined" ? document : null));
    const isTextEntryElement = requireFunction(options.isTextEntryElement, "isTextEntryElement");
    const isMobile = requireFunction(options.isMobile, "isMobile");
    const modalIsolationTargets = requireArray(options.modalIsolationTargets, "modalIsolationTargets");
    const isModalTargetOpen = requireFunction(options.isModalTargetOpen, "isModalTargetOpen");
    const addAppEvent = requireFunction(options.addAppEvent, "addAppEvent");
    const shellHints = requireArray(options.shellHints, "shellHints");

    let badgeContainer = null;
    let hintedTargets = new Map();

    function anyModalOpen() {
      return modalIsolationTargets.some(isModalTargetOpen);
    }

    function targetIsVisible(target) {
      if (!target || target.disabled || target.offsetParent === null) return false;
      const style = target.style || {};
      const view = documentTarget.defaultView || (typeof window !== "undefined" ? window : null);
      const computed = view && typeof view.getComputedStyle === "function" ? view.getComputedStyle(target) : null;
      const display = computed ? computed.display : style.display;
      const visibility = computed ? computed.visibility : style.visibility;
      if (display === "none" || visibility === "hidden") return false;
      // Must also be within the current viewport (not scrolled off-screen).
      // This prevents hint badges for file links that exist in the DOM but
      // are far above or below the visible conversation area.
      if (typeof target.getBoundingClientRect === "function") {
        const rect = target.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return false;
        const vh = view ? view.innerHeight : 0;
        const vw = view ? view.innerWidth : 0;
        if (rect.bottom < 0 || rect.top > vh || rect.right < 0 || rect.left > vw) return false;
      }
      return true;
    }

    function targetIsInsideOpenModal(target) {
      return modalIsolationTargets.some((modal) => isModalTargetOpen(modal) && modal && typeof modal.contains === "function" && modal.contains(target));
    }

    function collectTargets() {
      const targets = new Map();
      const sessionCards = Array.from(documentTarget.querySelectorAll("#sessions .session[data-session-id]"))
        .filter((card) => targetIsVisible(card) && !targetIsInsideOpenModal(card))
        .slice(0, SESSION_HINTS.length);
      for (const [index, card] of sessionCards.entries()) {
        targets.set(SESSION_HINTS[index], card);
      }
      for (const hint of shellHints) {
        const label = String(hint && hint.label || "").toLowerCase();
        const target = hint && hint.element;
        if (!label || label === "f" || targets.has(label) || !targetIsVisible(target) || targetIsInsideOpenModal(target)) continue;
        targets.set(label, target);
      }
      // Dynamic hints: assign available letters to clickable file references
      // in the conversation view (a[data-file-path] and a[data-file-picker-query]).
      // These are not locked — labels are assigned on the fly from whatever
      // single-char letters remain after sessions and shell hints.
      const ALPHABET = "abcdefghijklmnopqrstuvwxyz";
      const usedLabels = new Set(targets.keys());
      usedLabels.add("f"); // reserved leader
      const pool = ALPHABET.split("").filter((ch) => !usedLabels.has(ch));
      const fileLinks = Array.from(documentTarget.querySelectorAll(".chat a[data-file-path], .chat a[data-file-picker-query]"))
        .filter((link) => targetIsVisible(link) && !targetIsInsideOpenModal(link));
      for (const link of fileLinks) {
        if (!pool.length) break;
        targets.set(pool.shift(), link);
      }
      return targets;
    }

    function addBadge(target, label) {
      const badge = documentTarget.createElement("span");
      const rect = typeof target.getBoundingClientRect === "function" ? target.getBoundingClientRect() : { left: 0, top: 0 };
      badge.className = "codoxear-hint-badge";
      badge.textContent = label;
      Object.assign(badge.style, {
        position: "fixed",
        left: `${Math.max(0, Number(rect.left) || 0)}px`,
        top: `${Math.max(0, Number(rect.top) || 0)}px`,
        zIndex: "10000",
        pointerEvents: "none",
        padding: "1px 4px",
        borderRadius: "3px",
        background: "#111827",
        color: "#fff",
        font: "600 14px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace",
        boxShadow: "0 1px 3px rgba(0, 0, 0, .45)",
      });
      badgeContainer.appendChild(badge);
    }

    function enter() {
      if (hintedTargets.size || anyModalOpen() || isMobile()) return false;
      hintedTargets = collectTargets();
      if (!hintedTargets.size) return false;
      badgeContainer = documentTarget.createElement("div");
      badgeContainer.className = "codoxear-hint-mode";
      badgeContainer.setAttribute("aria-hidden", "true");
      badgeContainer.style.pointerEvents = "none";
      documentTarget.body.appendChild(badgeContainer);
      for (const [label, target] of hintedTargets) addBadge(target, label);
      return true;
    }

    function exit() {
      if (badgeContainer) {
        if (typeof badgeContainer.remove === "function") badgeContainer.remove();
        else if (badgeContainer.parentNode && typeof badgeContainer.parentNode.removeChild === "function") badgeContainer.parentNode.removeChild(badgeContainer);
      }
      badgeContainer = null;
      hintedTargets.clear();
    }

    function canEnter(target) {
      return !isMobile() && !anyModalOpen() && !isTextEntryElement(target) && !isTextEntryElement(documentTarget.activeElement);
    }

    function handleKeydown(event) {
      if (!hintedTargets.size) {
        if (event.defaultPrevented || event.key !== "f" || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
        if (!canEnter(event.target)) return;
        if (enter() && typeof event.preventDefault === "function") event.preventDefault();
        return;
      }

      if (event.key === "Escape" || event.key === "Backspace") {
        if (typeof event.preventDefault === "function") event.preventDefault();
        exit();
        return;
      }
      const label = String(event.key || "").toLowerCase();
      const target = hintedTargets.get(label);
      if (!target) {
        exit();
        return;
      }
      if (typeof event.preventDefault === "function") event.preventDefault();
      exit();
      // Prefer focus() for focusable form elements (textarea/input) so the
      // cursor lands in the field; fall back to click() for buttons.
      if (typeof target.focus === "function" && /^(textarea|input)$/i.test(target.tagName)) {
        try { target.focus({ preventScroll: false }); } catch (_) { target.click(); }
      } else {
        target.click();
      }
    }

    addAppEvent(documentTarget, "keydown", handleKeydown);

    return Object.freeze({
      enter,
      exit,
      collectTargets,
      canEnter,
      handleKeydown,
      isActive: () => hintedTargets.size > 0,
      dispose: exit,
    });
  }

  window.CodoxearHintMode = Object.freeze({ createHintModeController });
})();
