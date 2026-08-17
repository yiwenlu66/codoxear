
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

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`conversation copy dependency missing: ${name}`);
    return value;
  }

  // Conversation copy owns export access, stale-session rejection, formatter
  // authority, clipboard delivery, and all user feedback for this workflow.
  function createConversationCopyController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("conversation copy dependency missing: options");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function") throw new TypeError("conversation copy dependency missing: sessionState");
    const api = requireFunction(options.api, "api");
    const copyToClipboard = requireFunction(options.copyToClipboard, "copyToClipboard");
    const setToast = requireFunction(options.setToast, "setToast");
    const copyConversationFailureToast = requireFunction(options.copyConversationFailureToast, "copyConversationFailureToast");

    function successToast(messageCount) {
      return messageCount === 1 ? "Copied 1 message" : `Copied ${messageCount} messages`;
    }

    async function copyConversation() {
      const sid = sessionState.get("selected");
      if (!sid) return;
      try {
        const data = await api(`/api/sessions/${sid}/messages/export`);
        if (sessionState.get("selected") !== sid) return;
        const events = Array.isArray(data && data.events) ? data.events : [];
        const formatted = formatConversationForCopyResult(events);
        if (!formatted.text) {
          setToast("No conversation to copy");
          return;
        }
        await copyToClipboard(formatted.text);
        setToast(successToast(formatted.messageCount));
      } catch (err) {
        setToast(copyConversationFailureToast(err));
      }
    }

    return Object.freeze({ copyConversation });
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

export { createConversationCopyController, formatConversationForCopy, formatConversationForCopyResult, transcriptExportTooLargeCopyMessage };
