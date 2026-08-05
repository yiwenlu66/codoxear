(function (global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`pending user controller dependency missing: ${name}`);
    return value;
  }

  function createPendingUserController(options = {}) {
    const selectedSessionId = requireFunction(options.selectedSessionId, "selectedSessionId");
    const takePendingUserMatch = requireFunction(options.takePendingUserMatch, "takePendingUserMatch");
    const chatInner = options.chatInner;
    if (!chatInner || typeof chatInner.querySelector !== "function") throw new TypeError("pending user controller dependency missing: chatInner");
    const markdownHtml = requireFunction(options.markdownHtml, "markdownHtml");
    const time24 = requireFunction(options.time24, "time24");
    const rebuildDecorations = requireFunction(options.rebuildDecorations, "rebuildDecorations");
    const markEventSeen = requireFunction(options.markEventSeen, "markEventSeen");

    function consumePendingUserIfMatches(event, sessionId = selectedSessionId()) {
      const match = takePendingUserMatch(event, sessionId);
      if (!match) return false;
      const { id } = match;
      const pendingElement = chatInner.querySelector(`.msg.user[data-local-id="${id}"]`);
      if (!pendingElement) return false;

      pendingElement.style.opacity = "1";
      pendingElement.removeAttribute("data-local-id");
      pendingElement.removeAttribute("data-pending");

      const markdownElement = pendingElement.querySelector(".md");
      if (markdownElement && typeof event.text === "string") markdownElement.innerHTML = markdownHtml(event.text, sessionId);

      const row = pendingElement.closest(".msg-row");
      if (row && typeof event.ts === "number" && Number.isFinite(event.ts)) row.dataset.ts = String(event.ts);
      const timestamp = pendingElement.querySelector(".ts");
      if (timestamp && typeof event.ts === "number" && Number.isFinite(event.ts)) timestamp.textContent = time24(new Date(event.ts * 1000));
      rebuildDecorations({ preserveScroll: true });
      markEventSeen(event);
      return true;
    }

    return Object.freeze({ consumePendingUserIfMatches });
  }

  global.CodoxearPendingUser = Object.freeze({ createPendingUserController });
})(window);
