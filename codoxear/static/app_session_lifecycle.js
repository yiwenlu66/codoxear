

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`session lifecycle dependency missing: ${name}`);
    return value;
  }

  function createInterruptController(options = {}) {
    function requireInterruptFunction(value, name) {
      if (typeof value !== "function") throw new TypeError(`interrupt controller dependency missing: ${name}`);
      return value;
    }

    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function") throw new TypeError("interrupt controller dependency missing: sessionState");
    const setToast = requireInterruptFunction(options.setToast, "setToast");
    const api = requireInterruptFunction(options.api, "api");
    const now = requireInterruptFunction(options.now, "now");
    const setPollFastUntilMs = requireInterruptFunction(options.setPollFastUntilMs, "setPollFastUntilMs");
    const kickPoll = requireInterruptFunction(options.kickPoll, "kickPoll");

    async function interruptSelectedSession() {
      const sessionId = sessionState.get("selected");
      if (!sessionId) return;
      try {
        setToast("interrupting...");
        await api(`/api/sessions/${sessionId}/interrupt`, { method: "POST" });
        setPollFastUntilMs(now() + 2500);
        kickPoll(0);
      } catch (error) {
        setToast(`interrupt error: ${error.message}`);
      }
    }

    return Object.freeze({ interruptSelectedSession });
  }

  function createSessionLifecycleController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("session lifecycle dependency missing: options");
    const get = (name) => requireFunction(options[name], name);
    const asyncEpoch = options.asyncEpoch;
    if (!asyncEpoch || typeof asyncEpoch.currentGeneration !== "function" || typeof asyncEpoch.nextGeneration !== "function" || typeof asyncEpoch.incrementGeneration !== "function") {
      throw new TypeError("session lifecycle dependency missing: asyncEpoch");
    }
    const prepareSessionOpen = get("prepareSessionOpen");
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
    const syncAttachmentButton = get("syncAttachmentButton");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.set !== "function" || typeof sessionState.applyRuntime !== "function") {
      throw new TypeError("session lifecycle dependency missing: sessionState");
    }
    const resetChatRenderState = get("resetChatRenderState");
    const getSession = get("getSession");
    const isCurrent = get("isCurrent");
    const markClickLoad = get("markClickLoad");
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
    const invalidateOlderLoad = get("invalidateOlderLoad");
    const renderPendingTranscriptSlot = get("renderPendingTranscriptSlot");
    const applySessionRuntimeFromTail = get("applySessionRuntimeFromTail");
    const renderSessionTail = get("renderSessionTail");
    const replaceWith = requireFunction(
      Object.prototype.hasOwnProperty.call(options, "replaceWith") ? options.replaceWith : renderSessionTail,
      "replaceWith",
    );
    const openMessageEventSource = get("openMessageEventSource");
    const isMobile = get("isMobile");
    const closeSidebar = get("closeSidebar");
    const refreshFileCandidates = get("refreshFileCandidates");
    const isUnattendedOpen = get("isUnattendedOpen");
    const hideUnattendedMenu = get("hideUnattendedMenu");
    const saveSessionScrollPosition = get("saveSessionScrollPosition");
    const restoreSessionScrollPosition = get("restoreSessionScrollPosition");
    const clearSessionScrollPosition = get("clearSessionScrollPosition");
    const setActiveTranscriptPending = get("setActiveTranscriptPending");
    const deleteTranscriptSession = get("deleteTranscriptSession");
    const dropPendingUserRows = get("dropPendingUserRows");
    const sessionIdFromHash = get("sessionIdFromHash");
    const rememberPendingHashSession = get("rememberPendingHashSession");
    const sessionSelectable = get("sessionSelectable");
    const normalizeAgentBackendName = get("normalizeAgentBackendName");
    const providerChoiceToSettings = get("providerChoiceToSettings");
    const sessionCatalog = options.sessionCatalog;
    if (!sessionCatalog || typeof sessionCatalog.get !== "function") throw new TypeError("session lifecycle dependency missing: sessionCatalog");
    const backendSupportsFastForDefaults = get("backendSupportsFastForDefaults");
    const backendSupportsFast = (backend) => backendSupportsFastForDefaults(backend, sessionCatalog.get("newSessionDefaults"));
    const setToast = get("setToast");
    const confirmAction = get("confirmAction");
    const syncRecoveryUiForSession = get("syncRecoveryUiForSession");
    const sleep = get("sleep");
    const consoleError = get("consoleError");

    function clearSelectedSessionAfterRemoval(sessionId, { incrementPollGen = false, clearPollState = false } = {}) {
      if (sessionState.get("selected") !== sessionId) return false;
      handleFileViewerSessionUnavailable(sessionId);
      sessionState.set("selected", null);
      messageFlow().abortMessagePollRequest();
      if (incrementPollGen) asyncEpoch.incrementGeneration();
      if (clearPollState) messageFlow().clearPollSchedule();
      setActiveTranscriptPending();
      clearTranscriptForRemovedSession();
      sessionState.set("turnOpen", false);
      removePersistedSelected();
      setSessionHash("");
      sessionState.applyRuntime({ running: false, queueLen: 0, token: null, subagentsRunning: 0 });
      clearAttachments();
      syncAttachmentButton();
      resetChatRenderState();
      if (isUnattendedOpen()) hideUnattendedMenu();
      return true;
    }

    function clearDeletedSessionClientState(sessionId) {
      const selectedCleared = clearSelectedSessionAfterRemoval(sessionId);
      clearSessionScrollPosition(sessionId);
      deleteTranscriptSession(sessionId);
      dropPendingUserRows(sessionId);
      return selectedCleared;
    }

    async function openSession(sessionId, { useCache = true, fallbackToCacheOnFailure = false, forceRender = false } = {}) {
      const generation = asyncEpoch.nextGeneration();
      messageFlow().prepareSessionOpen();
      const oldSelected = sessionState.get("selected");
      const reloadingSelectedSession = oldSelected === sessionId;
      if (oldSelected && oldSelected !== sessionId) saveSessionScrollPosition(oldSelected);
      sessionState.set("selected", sessionId);
      if (oldSelected && oldSelected !== sessionId) saveComposerDraft(oldSelected);
      loadComposerDraft(sessionId);
      closeUnattendedForOtherSession(sessionId);
      persistSelected(sessionId);
      setSessionHash(sessionId);
      if (!reloadingSelectedSession) {
        resetTranscriptForSession();
      }
      syncAttachments();
      sessionState.applyRuntime({ running: false, queueLen: 0, token: null, subagentsRunning: 0 });
      if (!reloadingSelectedSession) resetChatRenderState();

      const session = getSession(sessionId);
      if (!isCurrent(sessionId, generation)) return null;
      markClickLoad();
      const optimisticBusy = Boolean(session && session.busy);
      const optimisticQueueLen = session && Number.isFinite(Number(session.queue_len)) ? Number(session.queue_len) : 0;
      const optimisticSubagentsRunning = session ? Math.max(0, Math.floor(Number(session.subagents_running) || 0)) : 0;
      sessionState.set("turnOpen", optimisticBusy);
      updateTypingStats(session, { updateSubagents: false });
      sessionState.applyRuntime({
        running: optimisticBusy,
        queueLen: optimisticQueueLen,
        token: session ? session.token || null : null,
        subagentsRunning: optimisticSubagentsRunning,
      });
      const fileViewerSyncStarted = beginFileViewerSync();

      const cachedTail = session ? getTailCache(sessionId) : null;
      let displayedCachedTail = false;
      if (useCache && session && cachedTail && tailCacheMatchesSession(cachedTail, session) && Array.isArray(cachedTail.events) && cachedTail.events.length) {
        applyCachedTail(sessionId, cachedTail, session);
        restoreSessionScrollPosition(sessionId);
        displayedCachedTail = true;
      }
      if (!displayedCachedTail && !reloadingSelectedSession) renderTranscriptLoading(sessionId);

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
          restoreSessionScrollPosition(sessionId);
          displayedCachedTail = true;
        }
        renderTranscriptLoadError(sessionId, error, { preserveTranscript: displayedCachedTail || reloadingSelectedSession });
        if (!isDisposed() && isCurrent(sessionId, generation)) kickPoll(messagePollDelayMs());
        return null;
      } finally {
        messageFlow().finishOpenSessionTailRequest(tailRequest);
      }
      if (!messageFlow().isCurrentOpenSessionTailRequest(tailRequest)) return null;
      messageFlow().markMessagePollSuccess();
      const slotChange = updateTranscriptSlot(sessionId, data);
      if (slotChange.ignoredStaleBound) {
        if (!reloadingSelectedSession) renderPendingTranscriptSlot(sessionId);
        applySessionRuntimeFromTail(sessionId, { transcript_state: "pending_bind", busy: data.busy, queue_len: data.queue_len, token: data.token });
        if (slotChange.current.state !== "failed") kickPoll(900);
        return data;
      }
      const tailEvents = Array.isArray(data.events) ? data.events : [];
      // A fresh bound snapshot whose key differs from the rendered
      // transcript's key is the authoritative replacement boundary: the
      // backend discarded the old working context (Pi /new, /resume, /fork,
      // a terminal-initiated renewal) and this tail fetch already holds the
      // replacement content, so replacing the DOM destroys nothing current.
      // The rendered key is the previous bound key on a direct rebind, or the
      // renewal marker's ignoredKey once beginTranscriptRenewal moved the
      // slot to pending_bind. Transient pending_bind snapshots without a
      // rendered key carry no such proof and keep the preserve path below.
      const renderedKey =
        slotChange.previous.state === "bound" ? slotChange.previous.key : slotChange.previous.ignoredKey;
      const transcriptReplaced = Boolean(
        slotChange.current.state === "bound" && renderedKey && slotChange.current.key !== renderedKey,
      );
      if (transcriptReplaced) {
        // Older-load state (in-flight page requests, cursors, cooldowns)
        // belongs to the discarded log; drop it before rendering the new tail.
        invalidateOlderLoad();
      }
      if (!reloadingSelectedSession || forceRender || transcriptReplaced) {
        // Fresh selections, explicit latest-tail requests, and transcript
        // replacements replace the DOM. Ordinary same-session reloads
        // preserve the rendered rows and scroll.
        if (slotChange.current.state === "bound" || slotChange.current.state === "failed") {
          replaceWith(tailEvents);
          // Replacement renders land at the new tail: restoring the old
          // transcript's saved scroll position would misplace the view.
          if (!forceRender && !transcriptReplaced) restoreSessionScrollPosition(sessionId);
        } else {
          renderPendingTranscriptSlot(sessionId);
        }
      }
      // Same-session reloads normally leave the DOM intact: appendEvent
      // (called by subsequent polls/SSE) deduplicates by message_id, so new
      // events appear without destroying existing content or scroll position.
      applySessionRuntimeFromTail(sessionId, data);
      if (slotChange.current.state !== "failed") { openMessageEventSource(sessionId, generation); kickPoll(900); }
      if (isMobile()) closeSidebar();
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
      if (sid === sessionState.get("selected")) {
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

    async function clearCommitUnknownSend(sid, previewText = "") {
      const sessionId = String(sid || "").trim();
      if (!sessionId) return false;
      const preview = String(previewText || "").trim();
      const suffix = preview ? `\n\nPrompt: ${preview.slice(0, 240)}${preview.length > 240 ? "..." : ""}` : "";
      const confirmed = await confirmAction({
        title: "Clear unknown-send marker?",
        message: `Clear the unknown-send marker only after checking the transcript or terminal. This does not undo a prompt that may already have been sent.${suffix}`,
        confirmText: "Clear marker",
        cancelText: "Cancel",
        destructive: true,
      });
      if (!confirmed) return false;
      try {
        await api(`/api/sessions/${sessionId}/commit_unknown_send/clear`, { method: "POST", body: {} });
        setToast("unknown send marker cleared");
        await refreshSessions();
        if (sessionState.get("selected") === sessionId) syncRecoveryUiForSession(sessionId);
        return true;
      } catch (error) {
        if (error && error.status === 401) {
          handleAuthLoss();
          return false;
        }
        setToast(`clear unknown send error: ${error && error.message ? error.message : "unknown error"}`);
        return false;
      }
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
      clearCommitUnknownSend,
      spawnSessionWithCwd,
    });
  }

export { createInterruptController, createSessionLifecycleController };
