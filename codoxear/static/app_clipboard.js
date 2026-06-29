(function () {
  "use strict";

  function copyTextViaSelection(text) {
    if (typeof document.execCommand !== "function") {
      throw new Error("Selection copy unavailable");
    }
    const value = String(text ?? "");
    const active = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const ta = document.createElement("textarea");
    ta.value = value;
    ta.setAttribute("aria-hidden", "true");
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "0";
    ta.style.left = "0";
    ta.style.width = "1px";
    ta.style.height = "1px";
    ta.style.padding = "0";
    ta.style.border = "0";
    ta.style.opacity = "0";
    ta.style.pointerEvents = "none";
    document.body.appendChild(ta);
    ta.focus({ preventScroll: true });
    ta.select();
    ta.setSelectionRange(0, value.length);
    const ok = document.execCommand("copy");
    ta.remove();
    if (active) active.focus({ preventScroll: true });
    if (!ok) throw new Error("Selection copy failed");
  }

  async function copyToClipboard(text) {
    const nav = typeof navigator !== "undefined" ? navigator : window.navigator;
    if (window.isSecureContext && nav && nav.clipboard && typeof nav.clipboard.writeText === "function") {
      await nav.clipboard.writeText(String(text ?? ""));
      return;
    }
    copyTextViaSelection(text);
  }

  window.CodoxearClipboard = Object.freeze({
    copyTextViaSelection,
    copyToClipboard,
  });
})();
