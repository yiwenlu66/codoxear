(function (global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`session lifecycle dependency missing: ${name}`);
    return value;
  }

  function createSessionLifecycleController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("session lifecycle dependency missing: options");
    const get = (name) => requireFunction(options[name], name);
    const nextPollGeneration = get("nextPollGeneration");
    const incrementPollGeneration = get("incrementPollGeneration");
    const prepareSessionOpen = get("prepareSessionOpen");
    const getSelected = get("getSelected");
    const setSelected = get("setSelected");
    const setActiveSession = get("setActiveSession");
    const saveComposerDraft = get("saveComposerDraft");
    const loadComposerDraft = get("loadComposerDraft");
    const closeUnattendedForOtherSession = get("closeUnattendedForOtherSession");
    const persistSelected = get("persistSelected");
    const removePersistedSelected = get("removePersistedSelected");
    const setSessionHash = get("setSessionHash");
    const resetTranscriptForSession = get("resetTranscriptForSession");
    const clearTranscriptForRemovedSession = get("clearTranscriptForRemovedSession");
    const syncAttachments = get("syncAttachments");
    const clearAttachments = get("clearAttachments");
    const updateQueueBadge = get("updateQueueBadge");
    const setStatus = get("setStatus");
    const setContext = get("setContext");
    const setTyping = get("setTyping");
    const resetChatRenderState = get("resetChatRenderState");
    const getSession = get("getSession");
    const isCurrent = get("isCurrent");
    const setTitle = get("setTitle");
    const setNoSessionTitle = get("setNoSessionTitle");
    const markClickLoad = get("markClickLoad");
    const setTurnOpen = get("setTurnOpen");
    const updateTypingStats = get("updateTypingStats");
    const beginFileViewerSync = get("beginFileViewerSync");
    const finishFileViewerSync = get("finishFileViewerSync");
    const handleFileViewerSessionUnavailable = get("handleFileViewerSessionUnavailable");
    const getTailCache = get("getTailCache");
    const tailCacheMatchesSession = get("tailCacheMatchesSession");
    const applyCachedTail = get("applyCachedTail");
    const renderTranscriptLoading = get("renderTranscriptLoading");
    const renderTranscriptLoadError = get("renderTranscriptLoadError");
    const messageFlow = get("messageFlow");
    const api = get("api");
    const initPageLimit = get("initPageLimit");
    const handleAuthLoss = get("handleAuthLoss");
    const refreshSessions = get("refreshSessions");
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
    const isUnattendedOpen = get("isUnattendedOpen");
    const hideUnattendedMenu = get("hideUnattendedMenu");
    const syncComposerSendButton = get("syncComposerSendButton");
    const syncQueueSubmitState = get("syncQueueSubmitState");
    const setActiveTranscriptPending = get("setActiveTranscriptPending");
    const deleteTranscriptSession = get("deleteTranscriptSession");
    const dropPendingUserRows = get("dropPendingUserRows");
    const sessionIdFromHash = get("sessionIdFromHash");
    const rememberPendingHashSession = get("rememberPendingHashSession");
    const sessionSelectable = get("sessionSelectable");
    const normalizeAgentBackendName = get("normalizeAgentBackendName");
    const providerChoiceToSettings = get("providerChoiceToSettings");
    const backendSupportsFast = get("backendSupportsFast");
    const setToast = get("setToast");
    const sleep = get("sleep");
    const consoleError = get("consoleError");

    function clearSelectedSessionAfterRemoval(sessionId, { incrementPollGen = false, clearPollState = false } = {}) {
      if (getSelected() !== sessionId) return false;
      handleFileViewerSessionUnavailable(sessionId);
      setSelected(null);
      messageFlow().abortMessagePollRequest();
      if (incrementPollGen) incrementPollGeneration();
      if (clearPollState) messageFlow().clearPollSchedule();
      setActiveTranscriptPending();
      clearTranscriptForRemovedSession();
      setTurnOpen(false);
      removePersistedSelected();
      setSessionHash("");
      setNoSessionTitle();
      setStatus({ running: false, queueLen: 0 });
      setContext(null);
      setTyping(false);
      clearAttachments();
      resetChatRenderState();
      updateQueueBadge();
      if (isUnattendedOpen()) hideUnattendedMenu();
      updateUnattendedButton();
      syncComposerSendButton();
      syncQueueSubmitState();
      return true;
    }

    function clearDeletedSessionClientState(sessionId) {
      const selectedCleared = clearSelectedSessionAfterRemoval(sessionId);
      deleteTranscriptSession(sessionId);
      dropPendingUserRows(sessionId);
      return selectedCleared;
    }

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
          clearSelectedSessionAfterRemoval(sessionId, { clearPollState: true });
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

    async function selectSession(id) {
      await openSession(id, { useCache: true });
    }

    async function selectSessionFromHash({ refreshIfMissing = false, deferIfMissing = false } = {}) {
      const sid = sessionIdFromHash();
      if (!sid) {
        rememberPendingHashSession("");
        return;
      }
      if (sid === getSelected()) {
        rememberPendingHashSession("");
        return;
      }
      let session = getSession(sid);
      if (!session && refreshIfMissing) {
        try {
          await refreshSessions();
        } catch (e) {
          if (e && e.status === 401) handleAuthLoss();
          else consoleError("hash session refresh failed", e);
          return;
        }
        session = getSession(sid);
      }
      if (!sessionSelectable(session)) {
        if (deferIfMissing) rememberPendingHashSession(sid);
        return;
      }
      rememberPendingHashSession("");
      await selectSession(sid);
    }

    async function spawnSessionWithCwd(cwd, resumeSessionId = null, worktreeBranch = null, sessionName = "", providerChoice = "chatgpt", model = "default", reasoningEffort = "high", fast = false, createInTmux = false, errorHandler = null, agentBackend = "codex") {
      if (!cwd || !String(cwd).trim()) {
        setToast("cwd unavailable");
        return null;
      }
      try {
        const backend = normalizeAgentBackendName(agentBackend);
        const modeLabel = resumeSessionId ? "resuming..." : worktreeBranch ? "creating worktree..." : createInTmux ? "starting in tmux..." : "starting...";
        const alias = String(sessionName || "").trim();
        const providerName = String(providerChoice || "").trim();
        const providerSettings = providerChoiceToSettings(providerName, backend);
        const modelName = String(model || "").trim();
        const effortName = String(reasoningEffort || "").trim().toLowerCase();
        setToast(modeLabel);
        const body = { cwd: String(cwd), agent_backend: backend };
        if (resumeSessionId) body.resume_session_id = String(resumeSessionId);
        if (worktreeBranch) body.worktree_branch = String(worktreeBranch);
        if (providerSettings.model_provider) body.model_provider = providerSettings.model_provider;
        if (providerSettings.preferred_auth_method) body.preferred_auth_method = providerSettings.preferred_auth_method;
        if (modelName) body.model = modelName;
        if (effortName) body.reasoning_effort = effortName;
        if (backendSupportsFast(backend) && fast) body.service_tier = "fast";
        if (createInTmux) body.create_in_tmux = true;
        const res = await api("/api/sessions", { method: "POST", body });
        if (res && res.pending && res.launch_id) {
          setToast(createInTmux ? "tmux session still starting" : "session still starting");
          await refreshSessions();
          return String(res.launch_id);
        }
        const brokerPid = res && res.broker_pid ? Number(res.broker_pid) : null;
        if (!brokerPid) {
          setToast("start failed");
          return null;
        }
        const doneLabel = resumeSessionId ? "resumed" : worktreeBranch ? "worktree started" : createInTmux ? "tmux started" : "started";
        setToast(`${doneLabel} (broker ${brokerPid})`);
        for (let i = 0; i < 60; i++) {
          const sessions = await refreshSessions();
          let found = (sessions || []).find((x) => Number(x.broker_pid || 0) === brokerPid);
          if (found) {
            if (alias && String(found.alias || "").trim() !== alias) {
              await api(`/api/sessions/${found.session_id}/rename`, { method: "POST", body: { name: alias } });
              const renamed = await refreshSessions();
              found = (renamed || []).find((x) => x.session_id === found.session_id) || found;
            }
            selectSession(found.session_id);
            return brokerPid;
          }
          await sleep(250);
        }
        setToast(`${doneLabel} session will appear once the agent writes its session log`);
        return brokerPid;
      } catch (e) {
        const errLabel = resumeSessionId ? "resume" : worktreeBranch ? "worktree start" : "start";
        if (typeof errorHandler === "function") errorHandler(e);
        setToast(`${errLabel} error: ${e.message}`);
        void refreshSessions().catch((err) => consoleError("refreshSessions failed after launch error", err));
        return null;
      }
    }

    return Object.freeze({
      openSession,
      selectSession,
      clearSelectedSessionAfterRemoval,
      selectSessionFromHash,
      clearDeletedSessionClientState,
      spawnSessionWithCwd,
    });
  }

  global.CodoxearSessionLifecycle = Object.freeze({ createSessionLifecycleController });
})(window);
