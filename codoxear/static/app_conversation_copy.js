(function () {
  "use strict";

  function formatConversationForCopy(events) {
    const parts = [];
    for (const ev of Array.isArray(events) ? events : []) {
      if (!ev || (ev.role !== "user" && ev.role !== "assistant")) continue;
      const text = String(ev.text || "").replace(/\s+$/g, "");
      if (!text.trim()) continue;
      const role = ev.role === "user" ? "User" : "Assistant";
      const ts = Number(ev.ts);
      const when = Number.isFinite(ts) ? ` (${new Date(ts * 1000).toLocaleString()})` : "";
      parts.push(`## ${role}${when}\n\n${text}`);
    }
    return parts.join("\n\n---\n\n").trim();
  }

  window.CodoxearConversationCopy = Object.freeze({
    formatConversationForCopy,
  });
})();
