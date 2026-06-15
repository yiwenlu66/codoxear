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
  });
})();
