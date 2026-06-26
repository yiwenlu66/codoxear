(function () {
  "use strict";

  function defaultButtonTooltip(attrs = {}, node = null) {
    const candidates = [attrs.title, attrs["aria-label"], attrs["data-tooltip"], attrs.text, node && node.textContent];
    for (const raw of candidates) {
      const value = String(raw || "").trim();
      if (value) return value;
    }
    return "";
  }

  function fmtTs(ts) {
    try {
      const d = new Date(ts * 1000);
      const y = String(d.getFullYear()).padStart(4, "0");
      const m = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      const hh = String(d.getHours()).padStart(2, "0");
      const mm = String(d.getMinutes()).padStart(2, "0");
      return `${y}-${m}-${day} ${hh}:${mm}`;
    } catch {
      return String(ts);
    }
  }

  function fmtBytes(n) {
    const v = Number(n);
    if (!Number.isFinite(v)) return String(n ?? "");
    if (v < 1024) return `${v} B`;
    const units = ["B", "KB", "MB", "GB", "TB"];
    let val = v;
    let u = 0;
    while (val >= 1024 && u < units.length - 1) {
      val /= 1024;
      u += 1;
    }
    const dec = val >= 100 ? 0 : val >= 10 ? 1 : 2;
    return `${val.toFixed(dec)} ${units[u]}`;
  }

  function baseName(p) {
    if (!p) return "";
    const s = String(p);
    const parts = s.split("/").filter(Boolean);
    return parts.length ? parts[parts.length - 1] : s;
  }

  function shortSessionId(sid) {
    const s = sid == null ? "" : String(sid);
    const m = s.match(/^([0-9a-f]{8})[0-9a-f-]{28}-(\d+)$/i);
    if (m) return `${m[1]}-${m[2]}`;
    return s.slice(0, 8);
  }

  function sessionDisplayName(s) {
    if (!s || typeof s !== "object") return "";
    const alias = typeof s.alias === "string" ? s.alias.trim() : "";
    if (alias) return alias;
    const cwdName = baseName(s.cwd);
    if (cwdName) return cwdName;
    const ts = typeof s.updated_ts === "number" && Number.isFinite(s.updated_ts)
      ? s.updated_ts
      : typeof s.start_ts === "number" && Number.isFinite(s.start_ts)
        ? s.start_ts
        : 0;
    return ts ? `Session ${fmtTs(ts)}` : "Session";
  }

  function fmtIdleAge(seconds) {
    const s = Number(seconds);
    if (!(s >= 0)) return "";
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.max(1, Math.floor(s / 60))}m`;
    if (s < 86400) return `${Math.max(1, Math.floor(s / 3600))}h`;
    return `${Math.max(1, Math.floor(s / 86400))}d`;
  }

  function fmtRelativeAge(seconds) {
    const base = fmtIdleAge(seconds);
    if (!base || base === "just now") return base;
    return `${base} ago`;
  }

  function sessionTitleWithId(s) {
    if (!s || typeof s !== "object") return "No session selected";
    const name = sessionDisplayName(s);
    return name || "No session selected";
  }

  function recoveryPromptPreview(text, maxLen = 320) {
    const raw = String(text || "").replace(/\s+/g, " ").trim();
    if (!raw) return "";
    return raw.length > maxLen ? `${raw.slice(0, maxLen)}…` : raw;
  }

  function fuzzyRecentCwdScore(candidate, query) {
    const text = String(candidate || "");
    const raw = String(query || "").trim().toLowerCase();
    if (!raw) return 0;
    const lower = text.toLowerCase();
    if (lower === raw) return 10000;
    const base = baseName(text).toLowerCase();
    if (base === raw) return 9000;
    let total = 0;
    for (const token of raw.split(/\s+/).filter(Boolean)) {
      const exactIdx = lower.indexOf(token);
      if (exactIdx >= 0) {
        const prev = exactIdx > 0 ? lower[exactIdx - 1] : "";
        const boundaryBonus = !prev || "/._-".includes(prev) ? 28 : 0;
        const baseIdx = base.indexOf(token);
        total += 260 - exactIdx * 2 + boundaryBonus + (baseIdx >= 0 ? 36 - baseIdx : 0);
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

  function compactChatSearchSnippet(text, query, limit = 96) {
    const clean = String(text || "").replace(/\s+/g, " ").trim();
    const maxLen = Math.max(24, Number(limit) || 96);
    if (!clean) return "";
    if (clean.length <= maxLen) return clean;
    const needle = String(query || "").trim().toLowerCase();
    let start = 0;
    if (needle) {
      const idx = clean.toLowerCase().indexOf(needle);
      if (idx > 24) start = Math.max(0, idx - 24);
    }
    const prefix = start > 0 ? "…" : "";
    const remaining = maxLen - prefix.length;
    const body = clean.slice(start, start + remaining);
    const suffix = start + remaining < clean.length ? "…" : "";
    return `${prefix}${body}${suffix}`;
  }

  function chatSearchTranscriptHint(match, query) {
    if (!match || typeof match !== "object") return "";
    const role = match.role === "user" ? "user" : match.role === "assistant" ? "assistant" : "match";
    const snippet = compactChatSearchSnippet(match.text, query);
    return snippet ? `${role}: ${snippet}` : "";
  }

  function iconSvg(name) {
    if (name === "menu")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>`;
    if (name === "refresh")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 12a8 8 0 1 1-2.34-5.66"/><path d="M20 4v6h-6"/></svg>`;
    if (name === "volume")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H3v6h3l5 4V5z"/><path d="M15 9a5 5 0 0 1 0 6"/><path d="M18.5 6.5a9 9 0 0 1 0 11"/></svg>`;
    if (name === "bell")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 17H5l2-2v-4a5 5 0 1 1 10 0v4l2 2h-4"/><path d="M10 17a2 2 0 0 0 4 0"/></svg>`;
    if (name === "play")
      return `<svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
    if (name === "unattended")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h3l2-4 3 8 2-4h6"/><path d="M12 21a9 9 0 1 0-9-9"/></svg>`;
    if (name === "stop")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>`;
    if (name === "plus")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>`;
    if (name === "logout")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 17l5-5-5-5"/><path d="M15 12H3"/><path d="M21 3v18"/></svg>`;
    if (name === "send")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>`;
    if (name === "paperclip")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-8.49 8.49a5 5 0 0 1-7.07-7.07l9.19-9.19a3.5 3.5 0 0 1 4.95 4.95l-9.19 9.19a2 2 0 0 1-2.83-2.83l8.49-8.49"/></svg>`;
    if (name === "down")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14"/><path d="M19 12l-7 7-7-7"/></svg>`;
    if (name === "up")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>`;
    if (name === "left")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5"/><path d="M12 5 5 12l7 7"/></svg>`;
    if (name === "right")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>`;
    if (name === "download")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>`;
    if (name === "save")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v6h8"/><path d="M9 20v-6h6v6"/></svg>`;
    if (name === "preview")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z"/><circle cx="12" cy="12" r="2.5"/></svg>`;
    if (name === "diff")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 4v16"/><path d="M17 4v16"/><path d="M4 7h6"/><path d="M14 17h6"/><path d="M14 7h6"/><path d="M4 17h6"/></svg>`;
    if (name === "chevronDown")
      return `<svg class="icon pickerChevronIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>`;
    if (name === "trash")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M6 6l1 16h10l1-16"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>`;
    if (name === "edit")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>`;
    if (name === "file")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/></svg>`;
    if (name === "x")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18"/><path d="M6 6l12 12"/></svg>`;
    if (name === "queue")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16"/><path d="M4 12h16"/><path d="M4 17h10"/></svg>`;
    if (name === "web")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><path d="M4 12h16"/><path d="M12 4a13 13 0 0 1 0 16"/><path d="M12 4a13 13 0 0 0 0 16"/></svg>`;
    if (name === "terminal")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m7 10 3 2-3 2"/><path d="M13 14h4"/></svg>`;
    if (name === "tmux")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 5v14"/><path d="M10 12h11"/><path d="m6 10 2 2-2 2"/></svg>`;
    if (name === "lightning")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 5 14h6l-1 8 8-12h-6z"/></svg>`;
    if (name === "duplicate")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="8" y="8" width="11" height="11" rx="2"/><rect x="5" y="5" width="11" height="11" rx="2"/></svg>`;
    if (name === "search")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>`;
    if (name === "copy")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M7 15H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v1"/></svg>`;
    if (name === "paste")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 4h6"/><path d="M10 2h4a1 1 0 0 1 1 1v2H9V3a1 1 0 0 1 1-1Z"/><rect x="6" y="5" width="12" height="16" rx="2"/><path d="m12 10 0 7"/><path d="m9.5 14.5 2.5 2.5 2.5-2.5"/></svg>`;
    if (name === "select")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V5h4"/><path d="M20 9V5h-4"/><path d="M4 15v4h4"/><path d="M20 15v4h-4"/><path d="M8 5H6a2 2 0 0 0-2 2v2"/><path d="M16 5h2a2 2 0 0 1 2 2v2"/><path d="M8 19H6a2 2 0 0 1-2-2v-2"/><path d="M16 19h2a2 2 0 0 0 2-2v-2"/></svg>`;
    if (name === "help")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.35 9.25a2.85 2.85 0 1 1 5.3 1.4c-.55.88-1.46 1.34-2.15 1.83-.74.53-1.25 1.08-1.25 2.02"/><circle cx="12" cy="17.2" r="0.9" fill="currentColor" stroke="none"/></svg>`;
    if (name === "info")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 10.5v5"/><circle cx="12" cy="7.6" r="0.9" fill="currentColor" stroke="none"/></svg>`;
    if (name === "settings")
      return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3.1"/><path d="M19.4 15a1 1 0 0 0 .2 1.1l.05.05a2 2 0 0 1-2.83 2.83l-.05-.05a1 1 0 0 0-1.1-.2 1 1 0 0 0-.6.91V20a2 2 0 1 1-4 0v-.08a1 1 0 0 0-.66-.94 1 1 0 0 0-1.08.23l-.05.05a2 2 0 1 1-2.83-2.83l.05-.05a1 1 0 0 0 .2-1.1 1 1 0 0 0-.91-.6H4a2 2 0 1 1 0-4h.08a1 1 0 0 0 .94-.66 1 1 0 0 0-.23-1.08l-.05-.05a2 2 0 1 1 2.83-2.83l.05.05a1 1 0 0 0 1.1.2 1 1 0 0 0 .6-.91V4a2 2 0 1 1 4 0v.08a1 1 0 0 0 .66.94 1 1 0 0 0 1.08-.23l.05-.05a2 2 0 1 1 2.83 2.83l-.05.05a1 1 0 0 0-.2 1.1 1 1 0 0 0 .91.6H20a2 2 0 1 1 0 4h-.08a1 1 0 0 0-.94.66 1 1 0 0 0 .23 1.08z"/></svg>`;
    return "";
  }

  window.CodoxearDisplay = Object.freeze({
    defaultButtonTooltip,
    fmtTs,
    fmtBytes,
    baseName,
    shortSessionId,
    sessionDisplayName,
    fmtIdleAge,
    fmtRelativeAge,
    sessionTitleWithId,
    recoveryPromptPreview,
    fuzzyRecentCwdScore,
    compactChatSearchSnippet,
    chatSearchTranscriptHint,
    iconSvg,
  });
})();
