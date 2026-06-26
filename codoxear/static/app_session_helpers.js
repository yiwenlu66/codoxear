(function () {
  "use strict";

  const SESSION_SIDEBAR_GROUPS = Object.freeze([
    Object.freeze({ key: "review", label: "Needs review" }),
    Object.freeze({ key: "now", label: "Now" }),
    Object.freeze({ key: "waiting", label: "Waiting" }),
    Object.freeze({ key: "later", label: "Later" }),
  ]);

  function sessionLaunchFailed(s) {
    return !!(s && String(s.launch_state || "").trim().toLowerCase() === "failed");
  }

  function sessionLaunchPending(s) {
    const state = s && String(s.launch_state || "").trim().toLowerCase();
    return !!(state && state !== "failed");
  }

  function sessionLaunchKind(s) {
    if (sessionLaunchFailed(s)) return "failed";
    if (s && s.transport === "tmux") return "web_tmux";
    if (s && s.owned) return "web";
    return "terminal";
  }

  function sessionLaunchIcon(s) {
    const kind = sessionLaunchKind(s);
    if (kind === "failed") return "info";
    if (kind === "web_tmux") return "tmux";
    return kind === "web" ? "web" : "terminal";
  }

  function sessionNeedsReview(s) {
    return !!(s && (sessionLaunchFailed(s) || s.orphan_recovery || s.queue_recovery || s.commit_unknown_send));
  }

  function sessionSidebarGroupKey(s) {
    if (sessionNeedsReview(s)) return "review";
    if (s && s.blocked) return "waiting";
    if (s && s.snoozed) return "later";
    return "now";
  }

  function sidebarSessionEntries(sessions) {
    const buckets = new Map(SESSION_SIDEBAR_GROUPS.map((group) => [group.key, []]));
    for (const s of Array.isArray(sessions) ? sessions : []) {
      const key = sessionSidebarGroupKey(s);
      const bucket = buckets.get(key) || buckets.get("now");
      bucket.push(s);
    }
    const entries = [];
    for (const group of SESSION_SIDEBAR_GROUPS) {
      const items = buckets.get(group.key) || [];
      if (!items.length) continue;
      entries.push({ type: "header", key: group.key, label: group.label, count: items.length });
      for (const session of items) entries.push({ type: "session", session });
    }
    return entries;
  }

  function sidebarRenderSignature(entries, { selectedId = "", swipeActions = false } = {}) {
    return JSON.stringify({
      selectedId: String(selectedId || ""),
      swipeActions: Boolean(swipeActions),
      entries: (Array.isArray(entries) ? entries : []).map((entry) => {
        if (!entry || entry.type === "header") return ["header", entry && entry.key, entry && entry.label, Number(entry && entry.count) || 0];
        const session = entry.session && typeof entry.session === "object" ? entry.session : {};
        return ["session", session.session_id || "", session];
      }),
    });
  }

  function sessionSelectable(s) {
    return !!(s && !sessionLaunchPending(s));
  }

  function sessionIsFast(s) {
    return !!(s && typeof s.service_tier === "string" && s.service_tier.trim().toLowerCase() === "fast");
  }

  function diagnosticsProviderDisplay(d, backend) {
    if (!d || typeof d !== "object") return "-";
    if (backend === "pi") return typeof d.model_provider === "string" && d.model_provider.trim() ? d.model_provider.trim() : "-";
    if (backend === "cc") return "-";
    if (typeof d.provider_choice === "string" && d.provider_choice.trim()) return d.provider_choice.trim();
    if (typeof d.model_provider === "string" && d.model_provider.trim()) return d.model_provider.trim();
    return "-";
  }

  function diagnosticsCopyText(sessionId, rows) {
    const rowLines = [];
    let hasSessionRow = false;
    for (const row of rows || []) {
      if (!row || !row.length) continue;
      const label = String(row[0] || "").trim();
      const value = String(row[1] || "-").trim() || "-";
      if (!label) continue;
      if (label.toLowerCase() === "session") hasSessionRow = true;
      rowLines.push(`${label}: ${value}`);
    }
    const lines = ["Codoxear session details"];
    if (sessionId && !hasSessionRow) lines.push(`Session: ${sessionId}`);
    return lines.concat(rowLines).join("\n");
  }

  function normalizeQueueItems(data) {
    if (data && Array.isArray(data.items)) {
      return data.items
        .filter((item) => item && typeof item === "object")
        .map((item) => ({
          id: typeof item.id === "string" ? item.id : "",
          text: typeof item.text === "string" ? item.text : "",
          sending: !!item.sending,
          commitUnknown: !!item.commit_unknown,
          orphanRecovery: !!item.orphan_recovery,
        }))
        .filter((item) => item.id && item.text.trim());
    }
    if (data && Array.isArray(data.queue)) {
      return data.queue
        .filter((text) => typeof text === "string" && text.trim())
        .map((text, idx) => ({ id: `legacy-${idx}`, text, sending: false, commitUnknown: false, orphanRecovery: false }));
    }
    return [];
  }

  window.CodoxearSessionHelpers = Object.freeze({
    SESSION_SIDEBAR_GROUPS,
    sessionLaunchFailed,
    sessionLaunchPending,
    sessionLaunchKind,
    sessionLaunchIcon,
    sessionNeedsReview,
    sessionSidebarGroupKey,
    sidebarSessionEntries,
    sidebarRenderSignature,
    sessionSelectable,
    sessionIsFast,
    diagnosticsProviderDisplay,
    diagnosticsCopyText,
    normalizeQueueItems,
  });
})();
