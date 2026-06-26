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

  window.CodoxearFileHelpers = Object.freeze({
    listFromFilesField,
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
  });
})();
