(function () {
  "use strict";

  async function copyToClipboard(text) {
    const nav = typeof navigator !== "undefined" ? navigator : window.navigator;
    if (!window.isSecureContext || !nav || !nav.clipboard || typeof nav.clipboard.writeText !== "function") {
      throw new Error("Clipboard API unavailable; requires a secure context (HTTPS)");
    }
    await nav.clipboard.writeText(String(text ?? ""));
  }

  window.CodoxearClipboard = Object.freeze({
    copyToClipboard,
  });
})();
