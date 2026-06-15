(function () {
  "use strict";

  const codoxearDisplay = window.CodoxearDisplay;
  if (!codoxearDisplay || typeof codoxearDisplay.fmtBytes !== "function") throw new Error("Codoxear display helpers failed to load");

  function listFromFilesField(val) {
    if (!Array.isArray(val)) return [];
    const out = [];
    for (const v of val) {
      if (typeof v !== "string") continue;
      const p = v;
      if (p === "" || out.includes(p)) continue;
      out.push(p);
    }
    return out;
  }

  function stripPathLocationSuffix(rawPath) {
    const raw = String(rawPath ?? "");
    const trimmed = raw.trim();
    let m = trimmed.match(/^(.*)#L(\d+)(?:-\d+)?$/);
    if (m) return m[1];
    m = trimmed.match(/^(.*):(\d+)(?::\d+)?$/);
    if (m && !/^[A-Za-z]:$/.test(m[1])) return m[1];
    return raw;
  }

  function isTextFileKind(kind) {
    return kind === "text" || kind === "markdown";
  }

  function isDiffableFileKind(kind) {
    return isTextFileKind(kind);
  }

  function blockedFileMessage(rel, reason, viewerMaxBytes, size) {
    const name = String(rel || "file");
    if (reason === "too_large") {
      const maxText = viewerMaxBytes ? codoxearDisplay.fmtBytes(viewerMaxBytes) : "the viewer limit";
      return `${name} is ${codoxearDisplay.fmtBytes(size)}. The viewer refuses to render text beyond ${maxText}. Use Download instead.`;
    }
    return `${name} is not renderable as text, markdown, image, or PDF. Use Download instead.`;
  }

  function formatPriorityOffset(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "0.00";
    return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;
  }

  window.CodoxearFileHelpers = Object.freeze({
    listFromFilesField,
    stripPathLocationSuffix,
    isTextFileKind,
    isDiffableFileKind,
    blockedFileMessage,
    formatPriorityOffset,
  });
})();
