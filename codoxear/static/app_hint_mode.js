
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
    let hintBuffer = "";

    function targetIsVisible(target) {
      if (!target || target.disabled || target.offsetParent === null) return false;
      const style = target.style || {};
      const view = documentTarget.defaultView || (typeof window !== "undefined" ? window : null);
      const computed = view && typeof view.getComputedStyle === "function" ? view.getComputedStyle(target) : null;
      const display = computed ? computed.display : style.display;
      const visibility = computed ? computed.visibility : style.visibility;
      if (display === "none" || visibility === "hidden") return false;
      // A target must have a visible footprint in the current viewport and
      // win at least one hit-test point. DOM visibility alone is insufficient:
      // a drawer or popover can leave an underlying control measurable while
      // making it impossible to activate.
      if (typeof target.getBoundingClientRect === "function") {
        const rect = target.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return false;
        const bounds = [rect.left, rect.top, rect.right, rect.bottom].map(Number);
        if (!bounds.every(Number.isFinite)) return true;
        const [rectLeft, rectTop, rectRight, rectBottom] = bounds;
        const vh = view ? Number(view.innerHeight) || 0 : 0;
        const vw = view ? Number(view.innerWidth) || 0 : 0;
        const left = Math.max(0, rectLeft);
        const top = Math.max(0, rectTop);
        const right = Math.min(vw, rectRight);
        const bottom = Math.min(vh, rectBottom);
        if (right <= left || bottom <= top) return false;
        if (typeof documentTarget.elementFromPoint === "function") {
          const insetX = Math.min(2, Math.max(0, (right - left) / 4));
          const insetY = Math.min(2, Math.max(0, (bottom - top) / 4));
          const points = [
            [(left + right) / 2, (top + bottom) / 2],
            [left + insetX, top + insetY],
            [right - insetX, top + insetY],
            [left + insetX, bottom - insetY],
            [right - insetX, bottom - insetY],
          ];
          const ownsHit = (hit) => hit === target || (typeof target.contains === "function" && target.contains(hit));
          if (!points.some(([x, y]) => ownsHit(documentTarget.elementFromPoint(x, y)))) return false;
        }
      }
      return true;
    }

    function targetIsInsideOpenModal(target) {
      return modalIsolationTargets.some((modal) => isModalTargetOpen(modal) && modal && typeof modal.contains === "function" && modal.contains(target));
    }

    function safeQueryAll(root, selector) {
      if (!root || typeof root.querySelectorAll !== "function") return [];
      try {
        return Array.from(root.querySelectorAll(selector));
      } catch (_) {
        return [];
      }
    }

    function controlLabel(target) {
      if (!target) return "";
      const labels = [
        typeof target.getAttribute === "function" ? target.getAttribute("aria-label") : "",
        typeof target.getAttribute === "function" ? target.getAttribute("title") : "",
        target.textContent,
        typeof target.getAttribute === "function" ? target.getAttribute("id") : "",
      ];
      return labels.map((value) => String(value || "").trim()).find(Boolean) || "control";
    }

    function targetIsHintExcluded(target) {
      return Boolean(target && typeof target.hasAttribute === "function" && target.hasAttribute("data-hint-excluded"));
    }

    function visibleInteractiveTargets(root) {
      return safeQueryAll(root, "button, input, textarea, select, [role='button'], [role='option'], a[href]")
        .filter((target) => targetIsVisible(target) && !target.disabled && !targetIsHintExcluded(target));
    }

    function fallbackHint(index) {
      const alphabet = "abcdefghijklmnopqrstuvwxyz";
      const high = Math.floor(index / alphabet.length) % alphabet.length;
      const low = index % alphabet.length;
      // `f` is reserved only while mode is inactive, so it provides a
      // prefix-safe namespace for controls beyond the one-key shell map.
      return `f${alphabet[high]}${alphabet[low]}`;
    }

    function assignAvailableHint(targets, target, usedLabels, pool, fallbackIndex) {
      if (!target || Array.from(targets.values()).includes(target)) return fallbackIndex;
      const label = controlLabel(target).toLowerCase();
      for (const candidate of label) {
        if (!/[a-z]/.test(candidate) || candidate === "f" || usedLabels.has(candidate)) continue;
        targets.set(candidate, target);
        usedLabels.add(candidate);
        const index = pool.indexOf(candidate);
        if (index >= 0) pool.splice(index, 1);
        return fallbackIndex;
      }
      const fallback = pool.shift();
      if (fallback) {
        targets.set(fallback, target);
        usedLabels.add(fallback);
        return fallbackIndex;
      }
      const overflow = fallbackHint(fallbackIndex);
      targets.set(overflow, target);
      usedLabels.add(overflow);
      return fallbackIndex + 1;
    }

    function openIsolationTarget() {
      return modalIsolationTargets.find((target) => isModalTargetOpen(target)) || null;
    }

    function collectTargets() {
      const targets = new Map();
      const isolationTarget = openIsolationTarget();
      if (isolationTarget) {
        // Dialogs/popovers are independent hint regions. Their controls get
        // letter hints even though the background shell is isolated.
        const controls = visibleInteractiveTargets(isolationTarget);
        const usedLabels = new Set(["f"]);
        const pool = "abcdefghijklmnopqrstuvwxyz".split("").filter((ch) => ch !== "f");
        let fallbackIndex = 0;
        for (const control of controls) fallbackIndex = assignAvailableHint(targets, control, usedLabels, pool, fallbackIndex);
        return targets;
      }

      const sessionCards = Array.from(documentTarget.querySelectorAll("#sessions .session[data-session-id]"))
        .filter((card) => targetIsVisible(card) && !targetIsInsideOpenModal(card))
        .slice(0, SESSION_HINTS.length);
      for (const [index, card] of sessionCards.entries()) {
        targets.set(SESSION_HINTS[index], card);
      }
      const usedLabels = new Set(targets.keys());
      usedLabels.add("f"); // reserved leader
      const pool = "abcdefghijklmnopqrstuvwxyz".split("").filter((ch) => !usedLabels.has(ch));
      for (const hint of shellHints) {
        const label = String(hint && hint.label || "").toLowerCase();
        const target = hint && hint.element;
        if (!label || label === "f" || targets.has(label) || !targetIsVisible(target) || targetIsInsideOpenModal(target)) continue;
        targets.set(label, target);
        usedLabels.add(label);
        const index = pool.indexOf(label);
        if (index >= 0) pool.splice(index, 1);
      }

      let fallbackIndex = 0;
      // Focused controllers and transcript rows are discovered at runtime so a
      // new interactive control cannot silently become keyboard-inaccessible.
      for (const control of visibleInteractiveTargets(documentTarget.body)) {
        if (targetIsInsideOpenModal(control)) continue;
        fallbackIndex = assignAvailableHint(targets, control, usedLabels, pool, fallbackIndex);
      }

      // Dynamic hints: assign available labels to clickable file references
      // in the conversation view (a[data-file-path] and a[data-file-picker-query]).
      const fileLinks = Array.from(documentTarget.querySelectorAll(".chat a[data-file-path], .chat a[data-file-picker-query]"))
        .filter((link) => targetIsVisible(link) && !targetIsInsideOpenModal(link));
      for (const link of fileLinks) fallbackIndex = assignAvailableHint(targets, link, usedLabels, pool, fallbackIndex);
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
        borderRadius: "0",
        border: "1px solid #141111",
        background: "#141111",
        color: "#fff",
        font: "600 14px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
      });
      badgeContainer.appendChild(badge);
    }

    function enter() {
      if (hintedTargets.size || isMobile()) return false;
      hintedTargets = collectTargets();
      hintBuffer = "";
      if (!hintedTargets.size) return false;
      badgeContainer = documentTarget.createElement("div");
      badgeContainer.className = "codoxear-hint-mode";
      badgeContainer.setAttribute("aria-hidden", "true");
      badgeContainer.style.pointerEvents = "none";
      // Native showModal() dialogs (settings, voice, edit, file viewer) live in
      // the top layer, which paints above any z-index in the document. The
      // badge layer joins the top layer as a manual popover so dialog hints
      // stay visible; popover positioning is not transformed, so the badges'
      // fixed viewport coordinates stay correct. Without popover support the
      // layer falls back to the document (and stays under open modals).
      if (typeof badgeContainer.showPopover === "function") {
        badgeContainer.setAttribute("popover", "manual");
        Object.assign(badgeContainer.style, {
          inset: "auto", margin: "0", padding: "0", border: "0",
          background: "transparent", overflow: "visible", pointerEvents: "none",
        });
      }
      documentTarget.body.appendChild(badgeContainer);
      if (typeof badgeContainer.showPopover === "function") {
        try { badgeContainer.showPopover(); } catch (_) { /* already showing */ }
      }
      for (const [label, target] of hintedTargets) addBadge(target, label);
      return true;
    }

    function exit() {
      if (badgeContainer) {
        if (typeof badgeContainer.hidePopover === "function") {
          try { badgeContainer.hidePopover(); } catch (_) { /* not showing */ }
        }
        if (typeof badgeContainer.remove === "function") badgeContainer.remove();
        else if (badgeContainer.parentNode && typeof badgeContainer.parentNode.removeChild === "function") badgeContainer.parentNode.removeChild(badgeContainer);
      }
      badgeContainer = null;
      hintedTargets.clear();
      hintBuffer = "";
    }

    function canEnter(target) {
      return !isTextEntryElement(target) && !isTextEntryElement(documentTarget.activeElement);
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
      if (label.length !== 1) {
        exit();
        return;
      }
      const nextBuffer = hintBuffer + label;
      const matchingLabels = Array.from(hintedTargets.keys()).filter((candidate) => candidate.startsWith(nextBuffer));
      if (!matchingLabels.length) {
        exit();
        return;
      }
      hintBuffer = nextBuffer;
      const target = hintedTargets.get(hintBuffer);
      if (!target || matchingLabels.some((candidate) => candidate !== hintBuffer)) return;
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

export { createHintModeController };
