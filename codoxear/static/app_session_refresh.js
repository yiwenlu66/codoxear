

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`session refresh dependency missing: ${name}`);
    return value;
  }

  function createSessionRefreshController(options = {}) {
    const api = requireFunction(options.api, "api");
    const isDisposed = requireFunction(options.isDisposed, "isDisposed");
    const apiResponseNotModified = requireFunction(options.apiResponseNotModified, "apiResponseNotModified");
    const sessionCatalog = options.sessionCatalog;
    if (!sessionCatalog || typeof sessionCatalog.get !== "function" || typeof sessionCatalog.applySnapshot !== "function") {
      throw new TypeError("session refresh dependency missing: sessionCatalog");
    }
    const emptyDefaults = requireFunction(options.emptyDefaults, "emptyDefaults");
    const clearFileDiscoveryCaches = requireFunction(options.clearFileDiscoveryCaches, "clearFileDiscoveryCaches");
    const useDesktopSessionActions = requireFunction(options.useDesktopSessionActions, "useDesktopSessionActions");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function") throw new TypeError("session refresh dependency missing: sessionState");    const clearSelectedSessionAfterRemoval = requireFunction(options.clearSelectedSessionAfterRemoval, "clearSelectedSessionAfterRemoval");
    const applySessionListTranscriptIdentity = requireFunction(options.applySessionListTranscriptIdentity, "applySessionListTranscriptIdentity");
    const syncAttachments = requireFunction(options.syncAttachments, "syncAttachments");
    const clearAttachments = requireFunction(options.clearAttachments, "clearAttachments");
    const renderSessions = requireFunction(options.renderSessions, "renderSessions");
    const hasDeferredRefresh = requireFunction(options.hasDeferredRefresh, "hasDeferredRefresh");
    const updateTypingStats = requireFunction(options.updateTypingStats, "updateTypingStats");
    const maybeSelectPendingHashSession = requireFunction(options.maybeSelectPendingHashSession, "maybeSelectPendingHashSession");

    let refreshInFlight = null;
    let refreshQueued = false;

    async function refreshSessions() {
      if (refreshInFlight) {
        refreshQueued = true;
        return refreshInFlight;
      }
      refreshInFlight = (async () => {
        let result = sessionCatalog.get("latestSessions");
        try {
          do {
            refreshQueued = false;
            result = await refreshSessionsOnce();
          } while (refreshQueued && !isDisposed());
          return result;
        } finally {
          refreshInFlight = null;
        }
      })();
      return refreshInFlight;
    }

    async function refreshSessionsOnce() {
      const data = await api("/api/sessions");
      let latestSessions = sessionCatalog.get("latestSessions");
      if (isDisposed()) return latestSessions;
      const notModified = apiResponseNotModified(data);
      const firstLoadNeedsPopulation = notModified && latestSessions.length === 0 && Array.isArray(data.sessions) && data.sessions.length > 0;
      if (notModified && !hasDeferredRefresh() && !firstLoadNeedsPopulation) return latestSessions;
      if (!notModified || firstLoadNeedsPopulation) {
        latestSessions = Array.isArray(data.sessions) ? data.sessions.slice() : [];
        sessionCatalog.applySnapshot({
          latestSessions,
          newSessionDefaults:
            data && typeof data.new_session_defaults === "object" && data.new_session_defaults
              ? data.new_session_defaults
              : emptyDefaults(),
          tmuxAvailable: Boolean(data.tmux_available),
          recentCwds: Array.isArray(data.recent_cwds)
            ? data.recent_cwds.filter((cwd, index, values) => typeof cwd === "string" && cwd.trim() && values.indexOf(cwd) === index)
            : [],
        });
        clearFileDiscoveryCaches();
      }
      const sessions = latestSessions.slice().sort((a, b) => {
        const priority = Number(b.final_priority || 0) - Number(a.final_priority || 0);
        if (priority) return priority;
        const updated = Number(b.updated_ts || b.start_ts || 0) - Number(a.updated_ts || a.start_ts || 0);
        if (updated) return updated;
        const started = Number(b.start_ts || 0) - Number(a.start_ts || 0);
        if (started) return started;
        return String(a.session_id || "").localeCompare(String(b.session_id || ""));
      });
      const sessionIndex = sessionCatalog.get("sessionIndex");
      let selected = sessionState.get("selected");
      if (selected && !sessionIndex.has(selected)) clearSelectedSessionAfterRemoval(selected);
      selected = sessionState.get("selected");
      if (selected) applySessionListTranscriptIdentity(selected, sessionIndex.get(selected));
      if (selected) syncAttachments();
      else clearAttachments();
      const selectedSession = selected ? sessionIndex.get(selected) : null;
      // The selected session's runtime snapshot is authoritative even when the
      // transcript identity is unchanged or a swipe defers sidebar rendering.
      // Runtime liveness is not an identity transition: every fresh listing
      // must be able to lower running while typing counters reconcile inside
      // updateTypingStats without regressing live monotonic counts.
      if (selectedSession) updateTypingStats(selectedSession);
      const renderedSidebar = renderSessions(sessions, { selectedId: selected, swipeActions: !useDesktopSessionActions() });
      if (!renderedSidebar) return sessions;
      maybeSelectPendingHashSession();
      return sessions;
    }

    return Object.freeze({ refreshSessions });
  }

export { createSessionRefreshController };
