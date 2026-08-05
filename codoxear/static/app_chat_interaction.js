/* Chat interaction composition: transcript rendering, message history, and send lifecycle. */
(function installCodoxearChatInteraction(global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`chat interaction dependency missing: ${name}`);
    return value;
  }

  function createChatInteractionController(options = {}) {
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const getPollGeneration = requireFunction(options.getPollGeneration, "getPollGeneration");
    const getSessionIndex = requireFunction(options.getSessionIndex, "getSessionIndex");
    const getSessionLifecycleController = requireFunction(options.getSessionLifecycleController, "getSessionLifecycleController");
    const getSessionRefreshController = requireFunction(options.getSessionRefreshController, "getSessionRefreshController");
    const getSending = requireFunction(options.getSending, "getSending");
    const getCurrentRunning = requireFunction(options.getCurrentRunning, "getCurrentRunning");
    const getSessionEditController = requireFunction(options.getSessionEditController, "getSessionEditController");
    const getQueueController = requireFunction(options.getQueueController, "getQueueController");
    const { wiring, document, storageSetItem, storageRemoveItem, codoxearViewport, codoxearSessions,
      sessionsWrap, sidebarEmptyHint, el, iconSvg, sidebarRenderSignature, sidebarSessionEntries,
      sessionDisplayName, sessionLaunchFailed, sessionLaunchPending, redactedLaunchErrorText,
      fmtRelativeAge, sidebarEffortCode, sidebarModelText, baseName, sessionIsFast, agentBackendLogoPath,
      agentBackendDisplayName, sessionAgentBackend, sessionLaunchIcon, sessionLaunchLabel, confirmApp, api,
      setToast, sessionProviderChoice, renderStatusChip, queueViewer, refreshQueueViewer } = options;
    const transcriptModule = options.codoxearTranscriptRender || window.CodoxearTranscriptRender;
    const historyModule = options.codoxearMessageHistory || window.CodoxearMessageHistory;
    const sendModule = options.codoxearSendLifecycle || window.CodoxearSendLifecycle;
    if (!transcriptModule || typeof transcriptModule.createTranscriptRenderController !== "function")
      throw new Error("Codoxear transcript render controller failed to load");
    if (!historyModule || typeof historyModule.createMessageHistoryController !== "function")
      throw new Error("Codoxear message history controller failed to load");
    if (!sendModule || typeof sendModule.createSendLifecycleController !== "function")
      throw new Error("Codoxear send lifecycle controller failed to load");

    let historyController = null;
    let sendLifecycleController = null;
    let attachmentsController = null;
    const transcript = transcriptModule.createTranscriptRenderController({
      ...options,
      getHistoryController: () => historyController,
      getSendLifecycleController: () => sendLifecycleController,
      getAttachmentsController: () => attachmentsController,
    });
    historyController = historyModule.createMessageHistoryController({
      ...options,
      transcript,
      getSendLifecycleController: () => sendLifecycleController,
      getAttachmentsController: () => attachmentsController,
    });
    sendLifecycleController = sendModule.createSendLifecycleController({ ...options, transcript, history: historyController });
    attachmentsController = sendLifecycleController.attachmentsController;

    function isMobile() { return codoxearViewport.isMobile(); }
    function useDesktopSessionActions() { return codoxearViewport.useDesktopSessionActions(); }
    function useTouchFileEditorControls() { return codoxearViewport.useTouchFileEditorControls(); }
    function setSidebarOpen(open) {
      document.body.classList.toggle("sidebar-open", Boolean(open));
      (open ? storageSetItem : storageRemoveItem)("codexweb.sidebarOpen", open ? "1" : undefined);
    }
    function setSidebarCollapsed(collapsed) {
      document.body.classList.toggle("sidebar-collapsed", Boolean(collapsed));
      (collapsed ? storageSetItem : storageRemoveItem)("codexweb.sidebarCollapsed", collapsed ? "1" : undefined);
    }
    function clearCommitUnknownSend(sid, previewText = "") {
      return getSessionLifecycleController().clearCommitUnknownSend(sid, previewText);
    }
    function refreshSessions() { return getSessionRefreshController().refreshSessions(); }
    function updateQueueBadge() {
      const queueController = getQueueController();
      if (!queueController) return;
      queueController.updateQueueBadge();
      if (queueViewer && queueViewer.style.display === "flex") void refreshQueueViewer();
    }
    const sidebarController = codoxearSessions.createSessionsController(wiring.createSessionsOptions({
      sessionsWrap, sidebarEmptyHint, el, iconSvg, sidebarRenderSignature, sidebarSessionEntries,
      sessionDisplayName, sessionLaunchFailed, sessionLaunchPending, redactedLaunchErrorText,
      fmtRelativeAge, sidebarEffortCode, sidebarModelText, baseName, sessionIsFast, agentBackendLogoPath,
      agentBackendDisplayName, sessionAgentBackend, sessionLaunchIcon, sessionLaunchLabel,
      confirmAction: (dialog) => confirmApp(dialog), api,
      clearDeletedSessionClientState: (...args) => getSessionLifecycleController().clearDeletedSessionClientState(...args),
      refreshSessions, setToast, openEditSession: (sid) => getSessionEditController().openEditSession(sid),
      duplicateSession: async (session) => {
        const cwd = session && session.cwd && session.cwd !== "?" ? session.cwd : "";
        if (!cwd) return setToast("cwd unavailable");
        return getSessionLifecycleController().spawnSessionWithCwd(cwd, null, null, "", sessionProviderChoice(session),
          session && session.model ? session.model : "default", session && session.reasoning_effort ? session.reasoning_effort : "high",
          sessionIsFast(session), !!(session && session.transport === "tmux"), null, sessionAgentBackend(session));
      },
      selectSession: (...args) => getSessionLifecycleController().selectSession(...args), setSidebarOpen,
      now: () => Date.now(), performanceNow: () => performance.now(), consoleError: (...args) => console.error(...args),
    }));

    return Object.freeze({
      attachmentsController, messageFlowController: sendLifecycleController.messageFlowController, sidebarController,
      chatSearchController: transcript.chatSearchController, chatNavigationController: transcript.chatNavigationController,
      transcriptSlotRuntime: transcript.transcriptSlotRuntime, typingRowRuntime: transcript.typingRowRuntime,
      transcriptScrollRuntime: transcript.transcriptScrollRuntime, transcriptDomRuntime: transcript.transcriptDomRuntime,
      transcriptEventRuntime: transcript.transcriptEventRuntime, olderLoadRuntime: historyController.olderLoadRuntime,
      resetChatRenderState: transcript.resetChatRenderState, clearOlderLoadError: historyController.clearOlderLoadError,
      updateChatNavButtons: transcript.updateChatNavButtons, closeChatSearch: transcript.closeChatSearch,
      clearRenderedTranscriptRange: historyController.clearRenderedTranscriptRange, initPageLimit: transcript.initPageLimit,
      dropPendingUserRows: transcript.dropPendingUserRows, updateSessionTranscriptSlot: transcript.updateSessionTranscriptSlot,
      tailCacheMatchesSession: transcript.tailCacheMatchesSession, applySessionListTranscriptIdentity: transcript.applySessionListTranscriptIdentity,
      updateQueueBadge, updateTypingStatsFromSession: transcript.updateTypingStatsFromSession,
      setTyping: transcript.setTyping, messagePollDelayMs: sendLifecycleController.messagePollDelayMs,
      kickPoll: sendLifecycleController.kickPoll, setPollFastUntilMs: sendLifecycleController.setPollFastUntilMs,
      openMessageEventSource: sendLifecycleController.openMessageEventSource, isMobile, useDesktopSessionActions,
      useTouchFileEditorControls, setSidebarOpen, setSidebarCollapsed, clearCommitUnknownSend, refreshSessions,
      loadOlderMessages: historyController.loadOlderMessages, applySessionRuntimeFromTail: historyController.applySessionRuntimeFromTail,
      renderSessionTail: historyController.renderSessionTail, recoveryDetailsText: historyController.recoveryDetailsText,
      syncRecoveryUiForSession: historyController.syncRecoveryUiForSession,
      renderPendingTranscriptSlot: historyController.renderPendingTranscriptSlot,
      renderTranscriptLoading: historyController.renderTranscriptLoading, renderTranscriptLoadError: historyController.renderTranscriptLoadError,
      applyCachedTail: historyController.applyCachedTail, jumpToLatest: historyController.jumpToLatest,
      rememberPendingHashSession: historyController.rememberPendingHashSession,
      maybeSelectPendingHashSession: historyController.maybeSelectPendingHashSession, getSending, getCurrentRunning,
    });
  }

  global.CodoxearChatInteraction = Object.freeze({ createChatInteractionController });
})(window);
