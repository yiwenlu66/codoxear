(() => {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`code copy dependency missing: ${name}`);
    return value;
  }

  function closestElement(node, selector) {
    return node && typeof node.closest === "function" ? node.closest(selector) : null;
  }

  function codeTextForCopyButton(button) {
    const pre = closestElement(button, "pre");
    const code = pre && typeof pre.querySelector === "function" ? pre.querySelector("code") : null;
    return code ? String(code.textContent || "") : "";
  }

  function codeCopyButtonFromEvent(event) {
    const target = event && event.target;
    return closestElement(target, ".code-copy-btn");
  }

  function createCodeBlockCopyRuntime(deps = {}) {
    const copyToClipboard = requireFunction(deps.copyToClipboard, "copyToClipboard");
    const setToast = requireFunction(deps.setToast, "setToast");
    const setTimeoutFn = requireFunction(deps.setTimeout, "setTimeout");
    const clearTimeoutFn = typeof deps.clearTimeout === "function" ? deps.clearTimeout : null;
    const resetTimers = new WeakMap();

    async function copyCodeBlock(button) {
      const text = codeTextForCopyButton(button);
      const originalLabel = button && typeof button.getAttribute === "function" ? button.getAttribute("aria-label") || "Copy code" : "Copy code";
      const originalTitle = button && typeof button.getAttribute === "function" ? button.getAttribute("title") || "Copy code" : "Copy code";
      try {
        await copyToClipboard(text);
        if (button && button.classList) button.classList.add("copied");
        if (button && typeof button.setAttribute === "function") {
          button.setAttribute("aria-label", "Copied code");
          button.setAttribute("title", "Copied code");
        }
        setToast("Copied code");
        if (clearTimeoutFn && resetTimers.has(button)) clearTimeoutFn(resetTimers.get(button));
        const timer = setTimeoutFn(() => {
          if (button && button.classList) button.classList.remove("copied");
          if (button && typeof button.setAttribute === "function") {
            button.setAttribute("aria-label", originalLabel);
            button.setAttribute("title", originalTitle);
          }
          resetTimers.delete(button);
        }, 1200);
        resetTimers.set(button, timer);
      } catch (err) {
        setToast(`copy failed: ${err && err.message ? err.message : "unknown error"}`);
      }
    }

    function handleClick(event) {
      const button = codeCopyButtonFromEvent(event);
      if (!button) return false;
      if (event && typeof event.preventDefault === "function") event.preventDefault();
      if (event && typeof event.stopPropagation === "function") event.stopPropagation();
      void copyCodeBlock(button);
      return true;
    }

    return Object.freeze({
      handleClick,
      codeTextForCopyButton,
    });
  }

  window.CodoxearCodeCopy = Object.freeze({
    createCodeBlockCopyRuntime,
    codeCopyButtonFromEvent,
    codeTextForCopyButton,
  });
})();
