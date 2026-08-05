(function installCodoxearSessionRefresh(global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`session refresh dependency missing: ${name}`);
    return value;
  }

  function createSessionRefreshController(options = {}) {
    const api = requireFunction(options.api, "api");
    const isDisposed = requireFunction(options.isDisposed, "isDisposed");
    const apiResponseNotModified = requireFunction(options.apiResponseNotModified, "apiResponseNotModified");
    const getLatestSessions = requireFunction(options.getLatestSessions, "getLatestSessions");
    const setLatestSessions = requireFunction(options.setLatestSessions, "setLatestSessions");
    const setNewSessionDefaults = requireFunction(options.setNewSessionDefaults, "setNewSessionDefaults");
    const emptyDefaults = requireFunction(options.emptyDefaults, "emptyDefaults");
    const setTmuxAvailable = requireFunction(options.setTmuxAvailable, "setTmuxAvailable");
    const setRecentCwds = requireFunction(options.setRecentCwds, "setRecentCwds");
    const refreshNewSessionDefaults = requireFunction(options.refreshNewSessionDefaults, "refreshNewSessionDefaults");
    const clearFileDiscoveryCaches = requireFunction(options.clearFileDiscoveryCaches, "clearFileDiscoveryCaches");
    const useDesktopSessionActions = requireFunction(options.useDesktopSessionActions, "useDesktopSessionActions");
    const setSessionIndex = requireFunction(options.setSessionIndex, "setSessionIndex");
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const clearSelectedSessionAfterRemoval = requireFunction(options.clearSelectedSessionAfterRemoval, "clearSelectedSessionAfterRemoval");
    const applySessionListTranscriptIdentity = requireFunction(options.applySessionListTranscriptIdentity, "applySessionListTranscriptIdentity");
    const syncRecoveryUiForSession = requireFunction(options.syncRecoveryUiForSession, "syncRecoveryUiForSession");
    const syncAttachments = requireFunction(options.syncAttachments, "syncAttachments");
    const clearAttachments = requireFunction(options.clearAttachments, "clearAttachments");
    const renderSessions = requireFunction(options.renderSessions, "renderSessions");
    const hasDeferredRefresh = requireFunction(options.hasDeferredRefresh, "hasDeferredRefresh");
    const setTitle = requireFunction(options.setTitle, "setTitle");
    const sessionTitle = requireFunction(options.sessionTitle, "sessionTitle");
    const updateTypingStats = requireFunction(options.updateTypingStats, "updateTypingStats");
    const updateUnattendedButton = requireFunction(options.updateUnattendedButton, "updateUnattendedButton");
    const updateQueueBadge = requireFunction(options.updateQueueBadge, "updateQueueBadge");
    const syncComposerSendButton = requireFunction(options.syncComposerSendButton, "syncComposerSendButton");
    const syncQueueSubmitState = requireFunction(options.syncQueueSubmitState, "syncQueueSubmitState");
    const maybeSelectPendingHashSession = requireFunction(options.maybeSelectPendingHashSession, "maybeSelectPendingHashSession");

    let refreshInFlight = null;
    let refreshQueued = false;

    async function refreshSessions() {
      if (refreshInFlight) {
        refreshQueued = true;
        return refreshInFlight;
      }
      refreshInFlight = (async () => {
        let result = getLatestSessions();
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
      console.log("[session-refresh] before /api/sessions");
      const data = await api("/api/sessions");
      console.log("[session-refresh] after /api/sessions", data);
      let latestSessions = getLatestSessions();
      if (isDisposed()) return latestSessions;
      const notModified = apiResponseNotModified(data);
      const firstLoadNeedsPopulation = notModified && latestSessions.length === 0 && Array.isArray(data.sessions) && data.sessions.length > 0;
      if (notModified && !hasDeferredRefresh() && !firstLoadNeedsPopulation) return latestSessions;
      if (!notModified || firstLoadNeedsPopulation) {
        latestSessions = Array.isArray(data.sessions) ? data.sessions.slice() : [];
        setLatestSessions(latestSessions);
        setNewSessionDefaults(
          data && typeof data.new_session_defaults === "object" && data.new_session_defaults
            ? data.new_session_defaults
            : emptyDefaults()
        );
        setTmuxAvailable(Boolean(data.tmux_available));
        setRecentCwds(
          Array.isArray(data.recent_cwds)
            ? data.recent_cwds.filter((cwd, index, values) => typeof cwd === "string" && cwd.trim() && values.indexOf(cwd) === index)
            : []
        );
        refreshNewSessionDefaults();
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
      const sessionIndex = new Map();
      for (const session of sessions) sessionIndex.set(session.session_id, session);
      setSessionIndex(sessionIndex);
      let selected = getSelected();
      if (selected && !sessionIndex.has(selected)) clearSelectedSessionAfterRemoval(selected);
      selected = getSelected();
      if (selected) {
        applySessionListTranscriptIdentity(selected, sessionIndex.get(selected));
        syncRecoveryUiForSession(selected);
      }
      if (selected) syncAttachments();
      else clearAttachments();
      console.log("[session-refresh] before renderSessions", sessions.length);
      const renderedSidebar = renderSessions(sessions, { selectedId: selected, swipeActions: !useDesktopSessionActions() });
      console.log("[session-refresh] after renderSessions", renderedSidebar);
      if (!renderedSidebar) return sessions;
      if (selected) {
        const session = sessionIndex.get(selected);
        if (session) {
          setTitle(sessionTitle(session));
          updateTypingStats(session);
        }
      }
      updateUnattendedButton();
      updateQueueBadge();
      syncComposerSendButton();
      syncQueueSubmitState();
      maybeSelectPendingHashSession();
      return sessions;
    }

    return Object.freeze({ refreshSessions });
  }

  global.CodoxearSessionRefresh = Object.freeze({ createSessionRefreshController });
})(window);
