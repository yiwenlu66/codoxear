(function () {
  "use strict";

  function conversationCopyParts(events) {
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
    return parts;
  }

  function formatConversationForCopyResult(events) {
    const parts = conversationCopyParts(events);
    return {
      text: parts.join("\n\n---\n\n").trim(),
      messageCount: parts.length,
    };
  }

  function formatConversationForCopy(events) {
    return formatConversationForCopyResult(events).text;
  }

  function formatCopyLimitBytes(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return "";
    const mib = n / (1024 * 1024);
    if (mib >= 1) {
      const rounded = Number.isInteger(mib) ? String(mib) : mib.toFixed(1).replace(/\.0$/, "");
      return `${rounded} MiB`;
    }
    const kib = n / 1024;
    if (kib >= 1) {
      const rounded = Number.isInteger(kib) ? String(kib) : kib.toFixed(1).replace(/\.0$/, "");
      return `${rounded} KiB`;
    }
    return `${Math.round(n)} bytes`;
  }

  function transcriptExportTooLargeCopyMessage(err) {
    if (!err || Number(err.status) !== 413) return "";
    const obj = err.obj && typeof err.obj === "object" ? err.obj : null;
    if (!obj || !Object.prototype.hasOwnProperty.call(obj, "max_bytes")) return "";
    const text = String(obj.error || err.message || "").toLowerCase();
    const knownExportGuard =
      text.includes("transcript-export-too-large") ||
      text.includes("too large to export") ||
      (text.includes("transcript") && text.includes("too large") && text.includes("export"));
    if (!knownExportGuard) return "";
    const limit = formatCopyLimitBytes(obj.max_bytes);
    return `Conversation too large to copy${limit ? ` (max ${limit})` : ""}. Use search or copy a smaller range.`;
  }

  window.CodoxearConversationCopy = Object.freeze({
    formatConversationForCopy,
    formatConversationForCopyResult,
    transcriptExportTooLargeCopyMessage,
  });
})();
