(function () {
  "use strict";

  const codoxearDisplay = window.CodoxearDisplay;
  if (!codoxearDisplay || typeof codoxearDisplay.fmtBytes !== "function" || typeof codoxearDisplay.baseName !== "function") throw new Error("Codoxear display helpers failed to load");

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

  function listFromFileRecords(val) {
    // Structured counterpart to listFromFilesField: each entry is normalized to
    // { path, apiPath }. Legacy string entries (no token) yield apiPath="";
    // object entries carry the reversible api_path token so a raw-byte filename
    // rehydrated from session.files can be reopened. Both snake_case (server
    // wire format) and camelCase (in-memory remember()) keys are accepted.
    if (!Array.isArray(val)) return [];
    const out = [];
    const seen = new Set();
    for (const v of val) {
      let path = "";
      let apiPath = "";
      if (typeof v === "string") {
        path = v;
      } else if (v && typeof v === "object") {
        path = typeof v.path === "string" ? v.path : "";
        const rawApi = v.api_path != null ? v.api_path : v.apiPath;
        apiPath = typeof rawApi === "string" ? rawApi : "";
      }
      if (path === "") continue;
      const identity = apiPath ? `${path}\u0000${apiPath}` : path;
      if (seen.has(identity)) continue;
      seen.add(identity);
      out.push({ path, apiPath });
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

  function fileVideoPreviewErrorText(err) {
    const raw = err && err.message ? String(err.message) : String(err || "");
    return raw.trim() || "compatible video preview failed";
  }

  function fileSearchScore(candidate, query) {
    const text = String(candidate || "");
    const raw = String(query || "").trim().toLowerCase();
    if (!raw) return 0;
    const lower = text.toLowerCase();
    if (lower === raw) return 12000;
    const base = codoxearDisplay.baseName(text).toLowerCase();
    if (base === raw) return 10000;
    let total = 0;
    for (const token of raw.split(/\s+/).filter(Boolean)) {
      const exactIdx = lower.indexOf(token);
      if (exactIdx >= 0) {
        const prev = exactIdx > 0 ? lower[exactIdx - 1] : "";
        const boundaryBonus = !prev || "/._-".includes(prev) ? 24 : 0;
        const baseIdx = base.indexOf(token);
        total += 240 - exactIdx * 2 + boundaryBonus + (baseIdx >= 0 ? 44 - baseIdx : 0);
        continue;
      }
      let pos = -1;
      let first = -1;
      let last = -1;
      let consecutive = 0;
      let boundaries = 0;
      for (const ch of token) {
        pos = lower.indexOf(ch, pos + 1);
        if (pos < 0) return -1;
        if (first < 0) first = pos;
        if (last >= 0 && pos === last + 1) consecutive += 1;
        if (pos === 0 || "/._-".includes(lower[pos - 1] || "")) boundaries += 1;
        last = pos;
      }
      const span = last - first + 1;
      total += 120 - first - Math.max(0, span - token.length) * 4 + consecutive * 10 + boundaries * 8;
    }
    return total;
  }

  function normalizeDraftFilePath(raw) {
    let path = String(raw || "").trim().replace(/\\/g, "/");
    path = path.replace(/^(?:\.\/)+/, "");
    if (!path || path === "." || path.startsWith("/") || path.endsWith("/") || path.includes("\x00")) return "";
    const parts = path.split("/");
    if (!parts.length) return "";
    for (const part of parts) {
      if (!part || part === "." || part === "..") return "";
    }
    return parts.join("/");
  }

  function filePickerFoldedSearchText(text) {
    const value = String(text || "");
    let folded = "";
    const startMap = [];
    const endMap = [];
    for (let idx = 0; idx < value.length; ) {
      const codePoint = value.codePointAt(idx);
      const raw = String.fromCodePoint(codePoint);
      const nextIdx = idx + raw.length;
      const lower = raw.toLowerCase();
      for (let lowerIdx = 0; lowerIdx < lower.length; lowerIdx += 1) {
        startMap.push(idx);
        endMap.push(nextIdx);
      }
      folded += lower;
      idx = nextIdx;
    }
    return { folded, startMap, endMap };
  }

  function filePickerOriginalRangeForFolded(mapped, start, end) {
    const foldedStart = Math.max(0, Math.floor(Number(start) || 0));
    const foldedEnd = Math.max(foldedStart, Math.floor(Number(end) || 0));
    if (!mapped || foldedEnd <= foldedStart || foldedStart >= mapped.startMap.length) return null;
    const last = Math.min(mapped.endMap.length - 1, foldedEnd - 1);
    const originalStart = mapped.startMap[foldedStart];
    const originalEnd = mapped.endMap[last];
    if (!Number.isFinite(originalStart) || !Number.isFinite(originalEnd) || originalEnd <= originalStart) return null;
    return [originalStart, originalEnd];
  }

  function filePickerMatchRanges(text, query) {
    const mapped = filePickerFoldedSearchText(text);
    const folded = mapped.folded;
    const raw = filePickerFoldedSearchText(String(query || "").trim()).folded;
    if (!raw || !folded) return [];
    const ranges = [];
    for (const token of raw.split(/\s+/).filter(Boolean)) {
      const exactIdx = folded.indexOf(token);
      if (exactIdx >= 0) {
        const range = filePickerOriginalRangeForFolded(mapped, exactIdx, exactIdx + token.length);
        if (range) ranges.push(range);
        continue;
      }
      let pos = -1;
      const positions = [];
      for (const ch of token) {
        pos = folded.indexOf(ch, pos + 1);
        if (pos < 0) return [];
        positions.push([pos, pos + ch.length]);
      }
      for (const [start, end] of positions) {
        const range = filePickerOriginalRangeForFolded(mapped, start, end);
        if (range) ranges.push(range);
      }
    }
    ranges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const merged = [];
    for (const [start, end] of ranges) {
      if (!merged.length || start > merged[merged.length - 1][1]) merged.push([start, end]);
      else merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], end);
    }
    return merged;
  }

  function filePickerMatchRangesForQuery(text, query) {
    const rawRanges = filePickerMatchRanges(text, query);
    if (rawRanges.length) return rawRanges;
    const normalized = normalizeDraftFilePath(query);
    if (!normalized || normalized === String(query || "").trim()) return [];
    return filePickerMatchRanges(text, normalized);
  }

  function filePickerCandidateScore(path, query) {
    const rawScore = fileSearchScore(path, query);
    const normalized = normalizeDraftFilePath(query);
    if (!normalized || normalized === String(query || "")) return rawScore;
    return Math.max(rawScore, fileSearchScore(path, normalized));
  }

  function compareFilePickerEntries(a, b) {
    const scoreDiff = Number(b.score || 0) - Number(a.score || 0);
    if (scoreDiff) return scoreDiff;
    const pathDiff = String(a.path || "").localeCompare(String(b.path || ""));
    if (pathDiff) return pathDiff;
    const gitPathDiff = Number(Boolean(a.gitPath)) - Number(Boolean(b.gitPath));
    if (gitPathDiff) return gitPathDiff;
    return Number(b.changed) - Number(a.changed) || Number(b.added) - Number(a.added);
  }

  function normalizeFileCandidateSource(source) {
    const value = String(source || "").trim();
    return ["changed", "mentioned", "recent"].includes(value) ? value : "";
  }

  function filePickerSectionLabel(source) {
    if (source === "changed") return "Changed files";
    if (source === "mentioned") return "Mentioned in chat";
    if (source === "recent") return "Recently opened";
    return "";
  }

  function duplicateFilePickerPaths(entries) {
    const counts = new Map();
    for (const entry of Array.isArray(entries) ? entries : []) {
      if (!entry || entry.createNew) continue;
      const path = String(entry.path || "");
      if (!path) continue;
      counts.set(path, Number(counts.get(path) || 0) + 1);
    }
    const out = new Set();
    for (const [path, count] of counts.entries()) {
      if (count > 1) out.add(path);
    }
    return out;
  }

  function rawByteDuplicatePaths(entries) {
    // Among duplicate display paths, identify those that have at least one
    // tokenized (raw-byte / non-UTF) entry. A non-empty ``apiPath`` token is
    // the reversible channel for a raw-byte filename; its absence means the
    // entry is a literal/display-only path. A collision only needs a
    // distinguishing hint when the two identities are actually different
    // kinds (tokenized vs literal), so this set gates the qualifier below.
    const counts = new Map();
    const tokenized = new Set();
    for (const entry of Array.isArray(entries) ? entries : []) {
      if (!entry || entry.createNew) continue;
      const path = String(entry.path || "");
      if (!path) continue;
      counts.set(path, Number(counts.get(path) || 0) + 1);
      if (typeof entry.apiPath === "string" && entry.apiPath !== "") tokenized.add(path);
    }
    const out = new Set();
    for (const [path, count] of counts.entries()) {
      if (count > 1 && tokenized.has(path)) out.add(path);
    }
    return out;
  }

  function filePickerIdentityHint(entry, duplicatePaths, options) {
    const showSourceSections = Boolean(options && options.showSourceSections);
    if (!entry || entry.createNew) return "";
    const path = String(entry.path || "");
    const duplicated = duplicatePaths && duplicatePaths.has(path);
    // A raw-byte/literal collision: only attach a distinguishing qualifier when
    // this duplicated path actually has a tokenized sibling, so ordinary
    // duplicates (two literal entries) stay noise-free. Duck-typed (has())
    // rather than ``instanceof Set`` so it survives cross-realm vm harnesses.
    const optsTok = options ? options.tokenizedDuplicatePaths : null;
    const tokenizedCollisions = optsTok && typeof optsTok.has === "function" ? optsTok : null;
    const hasRawByteCollision = Boolean(duplicated && tokenizedCollisions && tokenizedCollisions.has(path));
    const isTokenized = typeof entry.apiPath === "string" && entry.apiPath !== "";
    const byteQualifier = hasRawByteCollision ? (isTokenized ? "non-UTF bytes" : "literal name") : "";
    if (entry.pendingSessionPath) return byteQualifier ? `current folder · ${byteQualifier}` : "current folder";
    if (entry.gitPath && (duplicated || !showSourceSections)) {
      const base = entry.changed ? "git root · changed" : "git root";
      return byteQualifier ? `${base} · ${byteQualifier}` : base;
    }
    if (!entry.gitPath && duplicated) return byteQualifier ? `current folder · ${byteQualifier}` : "current folder";
    return "";
  }

  function filePickerTitle(entry, hint = "") {
    const path = String(entry && entry.path || "");
    if (!hint) return path;
    return `${path} — ${hint}`;
  }

  function positionAfterInsertedText(start, text) {
    const value = String(text || "");
    if (!value) return { lineNumber: start.lineNumber, column: start.column };
    const parts = value.replace(/\r\n?/g, "\n").split("\n");
    if (parts.length === 1) {
      return { lineNumber: start.lineNumber, column: start.column + parts[0].length };
    }
    return { lineNumber: start.lineNumber + parts.length - 1, column: parts[parts.length - 1].length + 1 };
  }

  function fileEditorDeleteCommandForKey(key) {
    if (key === "backspace") return "deleteLeft";
    if (key === "delete") return "deleteRight";
    return "";
  }

  function attachmentSafeStem(name) {
    const s = String(name || "file");
    const base = s.split("/").pop() || s;
    const dot = base.lastIndexOf(".");
    return (dot > 0 ? base.slice(0, dot) : base).replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 80) || "file";
  }

  function attachmentExtensionLower(name) {
    const s = String(name || "");
    const dot = s.lastIndexOf(".");
    return dot >= 0 ? s.slice(dot + 1).toLowerCase() : "";
  }

  function attachmentIsLikelyHeic(file) {
    const t = String(file && file.type ? file.type : "").toLowerCase();
    const e = attachmentExtensionLower(file && file.name ? file.name : "");
    return t.includes("heic") || t.includes("heif") || e === "heic" || e === "heif";
  }

  function attachmentLooksLikeImage(file) {
    const t = String(file && file.type ? file.type : "").toLowerCase();
    if (t.startsWith("image/")) return true;
    const e = attachmentExtensionLower(file && file.name ? file.name : "");
    return ["png", "jpg", "jpeg", "webp", "gif", "bmp", "svg", "avif", "heic", "heif"].includes(e);
  }

  function bytesToBase64(bytes, btoaFunc) {
    let bin = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoaFunc(bin);
  }

  window.CodoxearFileHelpers = Object.freeze({
    listFromFilesField,
    listFromFileRecords,
    stripPathLocationSuffix,
    isTextFileKind,
    isDiffableFileKind,
    blockedFileMessage,
    formatPriorityOffset,
    fileVideoPreviewErrorText,
    fileSearchScore,
    normalizeDraftFilePath,
    filePickerFoldedSearchText,
    filePickerOriginalRangeForFolded,
    filePickerMatchRanges,
    filePickerMatchRangesForQuery,
    filePickerCandidateScore,
    compareFilePickerEntries,
    normalizeFileCandidateSource,
    filePickerSectionLabel,
    duplicateFilePickerPaths,
    rawByteDuplicatePaths,
    filePickerIdentityHint,
    filePickerTitle,
    positionAfterInsertedText,
    fileEditorDeleteCommandForKey,
    attachmentSafeStem,
    attachmentExtensionLower,
    attachmentIsLikelyHeic,
    attachmentLooksLikeImage,
    bytesToBase64,
  });
})();
