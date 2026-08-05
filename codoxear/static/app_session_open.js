(function (global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`session open controller dependency missing: ${name}`);
    return value;
  }

  function createSessionOpenController(options = {}) {
    const get = (name) => requireFunction(options[name], name);
    const nextPollGeneration = get("nextPollGeneration");
    const prepareSessionOpen = get("prepareSessionOpen");
    const getSelected = get("getSelected");
    const setSelected = get("setSelected");
    const setActiveSession = get("setActiveSession");
    const saveComposerDraft = get("saveComposerDraft");
    const loadComposerDraft = get("loadComposerDraft");
    const closeUnattendedForOtherSession = get("closeUnattendedForOtherSession");
    const persistSelected = get("persistSelected");
    const setSessionHash = get("setSessionHash");
    const resetTranscriptForSession = get("resetTranscriptForSession");
    const syncAttachments = get("syncAttachments");
    const updateQueueBadge = get("updateQueueBadge");
    const setStatus = get("setStatus");
    const setContext = get("setContext");
    const setTyping = get("setTyping");
    const resetChatRenderState = get("resetChatRenderState");
    const getSession = get("getSession");
    const isCurrent = get("isCurrent");
    const setTitle = get("setTitle");
    const markClickLoad = get("markClickLoad");
    const setTurnOpen = get("setTurnOpen");
    const updateTypingStats = get("updateTypingStats");
    const beginFileViewerSync = get("beginFileViewerSync");
    const finishFileViewerSync = get("finishFileViewerSync");
    const getTailCache = get("getTailCache");
    const tailCacheMatchesSession = get("tailCacheMatchesSession");
    const applyCachedTail = get("applyCachedTail");
    const renderTranscriptLoading = get("renderTranscriptLoading");
    const messageFlow = get("messageFlow");
    const api = get("api");
    const initPageLimit = get("initPageLimit");
    const handleAuthLoss = get("handleAuthLoss");
    const clearRemovedSession = get("clearRemovedSession");
    const refreshSessions = get("refreshSessions");
    const renderTranscriptLoadError = get("renderTranscriptLoadError");
    const isDisposed = get("isDisposed");
    const kickPoll = get("kickPoll");
    const messagePollDelayMs = get("messagePollDelayMs");
    const updateTranscriptSlot = get("updateTranscriptSlot");
    const renderPendingTranscriptSlot = get("renderPendingTranscriptSlot");
    const applySessionRuntimeFromTail = get("applySessionRuntimeFromTail");
    const renderSessionTail = get("renderSessionTail");
    const openMessageEventSource = get("openMessageEventSource");
    const isMobile = get("isMobile");
    const closeSidebar = get("closeSidebar");
    const updateUnattendedButton = get("updateUnattendedButton");
    const refreshFileCandidates = get("refreshFileCandidates");
    const consoleError = get("consoleError");

    async function openSession(sessionId, { useCache = true, fallbackToCacheOnFailure = false } = {}) {
      const generation = nextPollGeneration();
      messageFlow().prepareSessionOpen();
      const oldSelected = getSelected();
      setSelected(sessionId);
      setActiveSession(sessionId);
      if (oldSelected && oldSelected !== sessionId) saveComposerDraft(oldSelected);
      loadComposerDraft(sessionId);
      closeUnattendedForOtherSession(sessionId);
      persistSelected(sessionId);
      setSessionHash(sessionId);
      resetTranscriptForSession();
      syncAttachments();
      updateQueueBadge();
      setStatus({ running: false, queueLen: 0 });
      setContext(null);
      setTyping(false);
      resetChatRenderState();

      const session = getSession(sessionId);
      if (!isCurrent(sessionId, generation)) return null;
      setTitle(session, sessionId);
      markClickLoad();
      const optimisticBusy = Boolean(session && session.busy);
      const optimisticQueueLen = session && Number.isFinite(Number(session.queue_len)) ? Number(session.queue_len) : 0;
      setTurnOpen(optimisticBusy);
      setStatus({ running: optimisticBusy, queueLen: optimisticQueueLen });
      setContext(session ? session.token || null : null);
      updateTypingStats(session);
      setTyping(optimisticBusy);
      const fileViewerSyncStarted = beginFileViewerSync();

      const cachedTail = session ? getTailCache(sessionId) : null;
      let displayedCachedTail = false;
      if (useCache && session && cachedTail && tailCacheMatchesSession(cachedTail, session) && Array.isArray(cachedTail.events) && cachedTail.events.length) {
        applyCachedTail(sessionId, cachedTail, session);
        displayedCachedTail = true;
      }
      if (!displayedCachedTail) renderTranscriptLoading(sessionId);

      const tailRequest = messageFlow().beginOpenSessionTailRequest(sessionId, generation);
      let data;
      try {
        data = await api(`/api/sessions/${sessionId}/messages/tail?limit=${initPageLimit()}`, { signal: tailRequest.signal });
      } catch (error) {
        if (error && error.status === 401) { handleAuthLoss(); return null; }
        if (messageFlow().isOpenSessionTailAbortError(tailRequest, error) || !messageFlow().isCurrentOpenSessionTailRequest(tailRequest)) return null;
        messageFlow().markMessagePollFailure();
        if (error && error.status === 404) {
          clearRemovedSession(sessionId, { clearPollState: true });
          void refreshSessions().catch((refreshError) => {
            if (refreshError && refreshError.status === 401) handleAuthLoss();
            else consoleError("refreshSessions failed after session disappeared", refreshError);
          });
          return null;
        }
        if (fallbackToCacheOnFailure && !displayedCachedTail && !useCache && session && cachedTail && tailCacheMatchesSession(cachedTail, session) && Array.isArray(cachedTail.events) && cachedTail.events.length) {
          applyCachedTail(sessionId, cachedTail, session);
          displayedCachedTail = true;
        }
        renderTranscriptLoadError(sessionId, error, { preserveTranscript: displayedCachedTail });
        if (!isDisposed() && isCurrent(sessionId, generation)) kickPoll(messagePollDelayMs());
        return null;
      } finally {
        messageFlow().finishOpenSessionTailRequest(tailRequest);
      }
      if (!messageFlow().isCurrentOpenSessionTailRequest(tailRequest)) return null;
      messageFlow().markMessagePollSuccess();
      const slotChange = updateTranscriptSlot(sessionId, data);
      if (slotChange.ignoredStaleBound) {
        renderPendingTranscriptSlot(sessionId);
        applySessionRuntimeFromTail(sessionId, { transcript_state: "pending_bind", busy: data.busy, queue_len: data.queue_len, token: data.token });
        if (slotChange.current.state !== "failed") kickPoll(900);
        return data;
      }
      if (slotChange.current.state === "bound" || slotChange.current.state === "failed") renderSessionTail(Array.isArray(data.events) ? data.events : []);
      else renderPendingTranscriptSlot(sessionId);
      applySessionRuntimeFromTail(sessionId, data);
      if (slotChange.current.state !== "failed") { openMessageEventSource(sessionId, generation); kickPoll(900); }
      if (isMobile()) closeSidebar();
      updateUnattendedButton();
      finishFileViewerSync(sessionId, fileViewerSyncStarted, refreshFileCandidates);
      return data;
    }

    return Object.freeze({ openSession });
  }

  global.CodoxearSessionOpen = Object.freeze({ createSessionOpenController });
})(window);
