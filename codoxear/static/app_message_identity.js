(function () {
  "use strict";

  function normalizeTextForPendingMatch(s) {
    // Normalize common platform newline differences to improve pending->ack reconciliation.
    return String(s || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  }

  function pendingMatchKey(s) {
    // Codex log serialization can trim trailing whitespace/newlines; match on a slightly
    // normalized key to avoid duplicating the optimistic local echo bubble.
    const t = normalizeTextForPendingMatch(s);
    return t.replace(/[ \t]+$/gm, "").replace(/\s+$/, "");
  }

  function eventKey(ev) {
    if (!ev || (ev.role !== "user" && ev.role !== "assistant")) return "";
    const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : null;
    if (ts === null) return "";
    const tsMs = Math.round(ts * 1000);
    const text = typeof ev.text === "string" ? pendingMatchKey(ev.text) : "";
    return `${ev.role}|${tsMs}|${text}`;
  }

  function chatAssistantDedupeKey(ev) {
    if (!ev || ev.role !== "assistant") return "";
    const raw = typeof ev.text === "string" ? ev.text : "";
    const text = pendingMatchKey(raw).replace(/\s+/g, " ").trim();
    if (!text) return "";
    const messageClass = typeof ev.message_class === "string" ? ev.message_class : "";
    return `${messageClass}|${text}`;
  }

  window.CodoxearMessageIdentity = Object.freeze({
    normalizeTextForPendingMatch,
    pendingMatchKey,
    eventKey,
    chatAssistantDedupeKey,
  });
})();
