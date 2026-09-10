import * as CodoxearMessageHistory from "./app_message_history.js";
import * as CodoxearSendLifecycle from "./app_send_lifecycle.js";
import * as CodoxearTranscriptRender from "./app_transcript_render.js";


/* Chat interaction composition: transcript rendering, message history, and send lifecycle. */

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`chat interaction dependency missing: ${name}`);
    return value;
  }

  function createChatInteractionController(options = {}) {
    const currentGeneration = options.currentGeneration;
    if (typeof currentGeneration !== "function") {
      throw new TypeError("chat interaction dependency missing: currentGeneration");
    }
    const sessionCatalog = options.sessionCatalog;
    if (!sessionCatalog || typeof sessionCatalog.get !== "function") throw new TypeError("chat interaction dependency missing: sessionCatalog");
    const getSessionLifecycleController = requireFunction(options.getSessionLifecycleController, "getSessionLifecycleController");
    const getSessionRefreshController = requireFunction(options.getSessionRefreshController, "getSessionRefreshController");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.applyRuntime !== "function" || typeof sessionState.subscribe !== "function") {
      throw new TypeError("chat interaction dependency missing: sessionState");
    }
    const getSessionEditController = requireFunction(options.getSessionEditController, "getSessionEditController");
    const getQueueController = requireFunction(options.getQueueController, "getQueueController");
    const { wiring, document, storageSetItem, storageRemoveItem, codoxearViewport, codoxearSessions,
      sessionsWrap, sidebarEmptyHint, el, iconSvg, sidebarRenderSignature, sidebarSessionEntries,
      sessionDisplayName, sessionLaunchFailed, sessionLaunchPending, redactedLaunchErrorText,
      fmtRelativeAge, sidebarEffortCode, sidebarModelText, baseName, sessionIsFast, agentBackendLogoPath,
      agentBackendDisplayName, sessionAgentBackend, sessionLaunchIcon, sessionLaunchLabel, confirmApp, api,
      setToast, sessionProviderChoice, queueViewer, refreshQueueViewer } = options;
    const transcriptModule = options.codoxearTranscriptRender || CodoxearTranscriptRender;
    const historyModule = options.codoxearMessageHistory || CodoxearMessageHistory;
    const sendModule = options.codoxearSendLifecycle || CodoxearSendLifecycle;
    if (!transcriptModule || typeof transcriptModule.createTranscriptRenderController !== "function")
      throw new Error("Codoxear transcript render controller failed to load");
    if (!historyModule || typeof historyModule.createMessageHistoryController !== "function")
      throw new Error("Codoxear message history controller failed to load");
    if (!sendModule || typeof sendModule.createSendLifecycleController !== "function")
      throw new Error("Codoxear send lifecycle controller failed to load");

    let historyController = null;
    let sendLifecycleController = null;
    let attachmentsController = null;
    const transcript = transcriptModule.createTranscriptRenderController(wiring.createTranscriptRenderOptions({
      currentGeneration,
      sessionCatalog,
      getSessionLifecycleController: options.getSessionLifecycleController,
      getSessionRefreshController: options.getSessionRefreshController,
      sessionState,
      isAppDisposed: options.isAppDisposed,
      getSessionEditController: options.getSessionEditController,
      getQueueController: options.getQueueController,
      isFileViewerOpen: options.isFileViewerOpen,
      upgradeCandidateFileRefs: options.upgradeCandidateFileRefs,
      isMobile: () => isMobile(),
      chatNavRail: options.chatNavRail,
      chatEmptyState: options.chatEmptyState,
      refreshSessions: () => getSessionRefreshController().refreshSessions(),
      jumpToLatest: () => historyController.jumpToLatest(),
      getHistoryController: () => historyController,
      getSendLifecycleController: () => sendLifecycleController,
      getAttachmentsController: () => attachmentsController,
      $: options.$,
      ATTACH_UPLOAD_MAX_BYTES: options.ATTACH_UPLOAD_MAX_BYTES,
      AbortController: options.AbortController,
      CHAT_DOM_WINDOW: options.CHAT_DOM_WINDOW,
      CHAT_DOM_WINDOW_WITH_HISTORY_SLACK: options.CHAT_DOM_WINDOW_WITH_HISTORY_SLACK,
      EventSource: options.EventSource,
      INIT_PAGE_LIMIT: options.INIT_PAGE_LIMIT,
      Node: options.Node,
      OLDER_AUTO_COOLDOWN_MS: options.OLDER_AUTO_COOLDOWN_MS,
      OLDER_CANCEL_PX: options.OLDER_CANCEL_PX,
      OLDER_PAGE_LIMIT: options.OLDER_PAGE_LIMIT,
      OLDER_TOP_TRIGGER_PX: options.OLDER_TOP_TRIGGER_PX,
      addAppEvent: options.addAppEvent,
      agentBackendDisplayName: options.agentBackendDisplayName,
      agentBackendLogoPath: options.agentBackendLogoPath,
      api: options.api,
      appConfirm: options.appConfirm,
      attachBtn: options.attachBtn,
      b64FromBytes: options.b64FromBytes,
      baseName: options.baseName,
      bottomSentinel: options.bottomSentinel,
      chat: options.chat,
      chatInner: options.chatInner,
      chatMarkdownHtmlCached: options.chatMarkdownHtmlCached,
      chatSearchAllHintEl: options.chatSearchAllHintEl,
      chatSearchBar: options.chatSearchBar,
      chatSearchBtn: options.chatSearchBtn,
      chatSearchCloseBtn: options.chatSearchCloseBtn,
      chatSearchInput: options.chatSearchInput,
      chatSearchNextBtn: options.chatSearchNextBtn,
      chatSearchPrevBtn: options.chatSearchPrevBtn,
      chatSearchStatus: options.chatSearchStatus,
      chatTimeChip: options.chatTimeChip,
      codeBlockCopyRuntime: options.codeBlockCopyRuntime,
      codoxearAttachments: options.codoxearAttachments,
      codoxearCodeCopy: options.codoxearCodeCopy,
      codoxearDisplay: options.codoxearDisplay,
      codoxearMessageFlow: options.codoxearMessageFlow,
      codoxearModal: options.codoxearModal,
      codoxearNavigationPulse: options.codoxearNavigationPulse,
      codoxearPendingUser: options.codoxearPendingUser,
      codoxearSessions: options.codoxearSessions,
      codoxearViewport: options.codoxearViewport,
      composer: options.composer,
      confirmApp: options.confirmApp,
      copyToClipboard: options.copyToClipboard,
      dataTransferHasFiles: options.dataTransferHasFiles,
      diagViewer: options.diagViewer,
      document: options.document,
      editViewer: options.editViewer,
      el: options.el,
      extractFilesFromClipboardData: options.extractFilesFromClipboardData,
      extractFilesFromDropData: options.extractFilesFromDropData,
      fmtBytes: options.fmtBytes,
      fmtRelativeAge: options.fmtRelativeAge,
      handleAppAuthLoss: options.handleAppAuthLoss,
      helpViewer: options.helpViewer,
      iconSvg: options.iconSvg,
      imgInput: options.imgInput,
      isLikelyHeic: options.isLikelyHeic,
      isModalTargetOpen: options.isModalTargetOpen,
      isTextEntryElement: options.isTextEntryElement,
      jumpBtn: options.jumpBtn,
      looksLikeImage: options.looksLikeImage,
      modalIsolationTargets: options.modalIsolationTargets,
      navigator: options.navigator,
      networkStatus: options.networkStatus,
      newSessionDialogController: options.newSessionDialogController,
      nextUserBtn: options.nextUserBtn,
      olderBtn: options.olderBtn,
      olderError: options.olderError,
      olderErrorText: options.olderErrorText,
      olderWrap: options.olderWrap,
      performance: options.performance,
      prevUserBtn: options.prevUserBtn,
      pushPerfSample: options.pushPerfSample,
      queueViewer: options.queueViewer,
      redactedLaunchErrorText: options.redactedLaunchErrorText,
      refreshQueueViewer: options.refreshQueueViewer,
      requestAnimationFrame: options.requestAnimationFrame,
      resizeComposer: options.resizeComposer,
      resolveAppUrl: options.resolveAppUrl,
      safeAttachmentStem: options.safeAttachmentStem,
      sendChoice: options.sendChoice,
      sessionAgentBackend: options.sessionAgentBackend,
      sessionDisplayName: options.sessionDisplayName,
      sessionHasOrphanQueueRecovery: options.sessionHasOrphanQueueRecovery,
      sessionHasUnknownSend: options.sessionHasUnknownSend,
      sessionIdFromHash: options.sessionIdFromHash,
      sessionIsFast: options.sessionIsFast,
      sessionIsOrphanRecovery: options.sessionIsOrphanRecovery,
      sessionLaunchFailed: options.sessionLaunchFailed,
      sessionLaunchIcon: options.sessionLaunchIcon,
      sessionLaunchLabel: options.sessionLaunchLabel,
      sessionLaunchPending: options.sessionLaunchPending,
      sessionProviderChoice: options.sessionProviderChoice,
      sessionSelectable: options.sessionSelectable,
      sessionsWrap: options.sessionsWrap,
      setTimeout: options.setTimeout,
      setToast: options.setToast,
      sidebarEffortCode: options.sidebarEffortCode,
      sidebarEmptyHint: options.sidebarEmptyHint,
      sidebarModelText: options.sidebarModelText,
      sidebarRenderSignature: options.sidebarRenderSignature,
      sidebarSessionEntries: options.sidebarSessionEntries,
      storageRemoveItem: options.storageRemoveItem,
      storageSetItem: options.storageSetItem,
      textarea: options.textarea,
      window: options.window,
      wiring: options.wiring,
    }));
    historyController = historyModule.createMessageHistoryController(wiring.createMessageHistoryOptions({
      currentGeneration,
      sessionCatalog,
      getSessionLifecycleController: options.getSessionLifecycleController,
      getSessionRefreshController: options.getSessionRefreshController,
      getSendLifecycleController: () => sendLifecycleController,
      transcript: transcript,
      wiring: options.wiring,
      olderWrap: options.olderWrap,
      olderBtn: options.olderBtn,
      olderError: options.olderError,
      olderErrorText: options.olderErrorText,
      AbortController: options.AbortController,
      performance: options.performance,
      OLDER_AUTO_COOLDOWN_MS: options.OLDER_AUTO_COOLDOWN_MS,
      OLDER_PAGE_LIMIT: options.OLDER_PAGE_LIMIT,
      api: options.api,
      handleAppAuthLoss: options.handleAppAuthLoss,
      sessionState,
      sessionLaunchFailed: options.sessionLaunchFailed,
      confirmApp: options.confirmApp,
      setToast: options.setToast,
      codoxearDisplay: options.codoxearDisplay,
      redactedLaunchErrorText: options.redactedLaunchErrorText,
      sessionIdFromHash: options.sessionIdFromHash,
      sessionSelectable: options.sessionSelectable,
    }));
    sendLifecycleController = sendModule.createSendLifecycleController(wiring.createSendLifecycleOptions({
      currentGeneration,
      sessionCatalog,
      getSessionLifecycleController: options.getSessionLifecycleController,
      getSessionRefreshController: options.getSessionRefreshController,
      sessionState,
      isAppDisposed: options.isAppDisposed,
      getSessionEditController: options.getSessionEditController,
      getQueueController: options.getQueueController,
      isFileViewerOpen: options.isFileViewerOpen,
      upgradeCandidateFileRefs: options.upgradeCandidateFileRefs,
      transcript: transcript,
      history: historyController,
      $: options.$,
      ATTACH_UPLOAD_MAX_BYTES: options.ATTACH_UPLOAD_MAX_BYTES,
      AbortController: options.AbortController,
      CHAT_DOM_WINDOW: options.CHAT_DOM_WINDOW,
      CHAT_DOM_WINDOW_WITH_HISTORY_SLACK: options.CHAT_DOM_WINDOW_WITH_HISTORY_SLACK,
      EventSource: options.EventSource,
      INIT_PAGE_LIMIT: options.INIT_PAGE_LIMIT,
      Node: options.Node,
      OLDER_AUTO_COOLDOWN_MS: options.OLDER_AUTO_COOLDOWN_MS,
      OLDER_CANCEL_PX: options.OLDER_CANCEL_PX,
      OLDER_PAGE_LIMIT: options.OLDER_PAGE_LIMIT,
      OLDER_TOP_TRIGGER_PX: options.OLDER_TOP_TRIGGER_PX,
      addAppEvent: options.addAppEvent,
      agentBackendDisplayName: options.agentBackendDisplayName,
      agentBackendLogoPath: options.agentBackendLogoPath,
      api: options.api,
      appConfirm: options.appConfirm,
      attachBtn: options.attachBtn,
      b64FromBytes: options.b64FromBytes,
      baseName: options.baseName,
      bottomSentinel: options.bottomSentinel,
      chat: options.chat,
      chatInner: options.chatInner,
      chatMarkdownHtmlCached: options.chatMarkdownHtmlCached,
      chatSearchAllHintEl: options.chatSearchAllHintEl,
      chatSearchBar: options.chatSearchBar,
      chatSearchBtn: options.chatSearchBtn,
      chatSearchCloseBtn: options.chatSearchCloseBtn,
      chatSearchInput: options.chatSearchInput,
      chatSearchNextBtn: options.chatSearchNextBtn,
      chatSearchPrevBtn: options.chatSearchPrevBtn,
      chatSearchStatus: options.chatSearchStatus,
      chatTimeChip: options.chatTimeChip,
      codeBlockCopyRuntime: options.codeBlockCopyRuntime,
      codoxearAttachments: options.codoxearAttachments,
      codoxearCodeCopy: options.codoxearCodeCopy,
      codoxearDisplay: options.codoxearDisplay,
      codoxearMessageFlow: options.codoxearMessageFlow,
      codoxearModal: options.codoxearModal,
      codoxearNavigationPulse: options.codoxearNavigationPulse,
      codoxearPendingUser: options.codoxearPendingUser,
      codoxearSessions: options.codoxearSessions,
      codoxearViewport: options.codoxearViewport,
      composer: options.composer,
      confirmApp: options.confirmApp,
      copyToClipboard: options.copyToClipboard,
      dataTransferHasFiles: options.dataTransferHasFiles,
      diagViewer: options.diagViewer,
      document: options.document,
      editViewer: options.editViewer,
      el: options.el,
      extractFilesFromClipboardData: options.extractFilesFromClipboardData,
      extractFilesFromDropData: options.extractFilesFromDropData,
      fmtBytes: options.fmtBytes,
      fmtRelativeAge: options.fmtRelativeAge,
      handleAppAuthLoss: options.handleAppAuthLoss,
      helpViewer: options.helpViewer,
      iconSvg: options.iconSvg,
      imgInput: options.imgInput,
      isLikelyHeic: options.isLikelyHeic,
      isModalTargetOpen: options.isModalTargetOpen,
      isTextEntryElement: options.isTextEntryElement,
      jumpBtn: options.jumpBtn,
      looksLikeImage: options.looksLikeImage,
      modalIsolationTargets: options.modalIsolationTargets,
      navigator: options.navigator,
      networkStatus: options.networkStatus,
      newSessionDialogController: options.newSessionDialogController,
      nextUserBtn: options.nextUserBtn,
      olderBtn: options.olderBtn,
      olderError: options.olderError,
      olderErrorText: options.olderErrorText,
      olderWrap: options.olderWrap,
      performance: options.performance,
      prevUserBtn: options.prevUserBtn,
      pushPerfSample: options.pushPerfSample,
      queueViewer: options.queueViewer,
      redactedLaunchErrorText: options.redactedLaunchErrorText,
      refreshQueueViewer: options.refreshQueueViewer,
      requestAnimationFrame: options.requestAnimationFrame,
      resizeComposer: options.resizeComposer,
      resolveAppUrl: options.resolveAppUrl,
      safeAttachmentStem: options.safeAttachmentStem,
      sendChoice: options.sendChoice,
      sessionAgentBackend: options.sessionAgentBackend,
      sessionDisplayName: options.sessionDisplayName,
      sessionHasOrphanQueueRecovery: options.sessionHasOrphanQueueRecovery,
      sessionHasUnknownSend: options.sessionHasUnknownSend,
      sessionIdFromHash: options.sessionIdFromHash,
      sessionIsFast: options.sessionIsFast,
      sessionIsOrphanRecovery: options.sessionIsOrphanRecovery,
      sessionLaunchFailed: options.sessionLaunchFailed,
      sessionLaunchIcon: options.sessionLaunchIcon,
      sessionLaunchLabel: options.sessionLaunchLabel,
      sessionLaunchPending: options.sessionLaunchPending,
      sessionProviderChoice: options.sessionProviderChoice,
      sessionSelectable: options.sessionSelectable,
      sessionsWrap: options.sessionsWrap,
      setTimeout: options.setTimeout,
      setToast: options.setToast,
      sidebarEffortCode: options.sidebarEffortCode,
      sidebarEmptyHint: options.sidebarEmptyHint,
      sidebarModelText: options.sidebarModelText,
      sidebarRenderSignature: options.sidebarRenderSignature,
      sidebarSessionEntries: options.sidebarSessionEntries,
      storageRemoveItem: options.storageRemoveItem,
      storageSetItem: options.storageSetItem,
      textarea: options.textarea,
      window: options.window,
      wiring: options.wiring,
    }));
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
    const sidebarController = codoxearSessions.createSessionsController(wiring.createSessionsOptions({
      sessionState, sessionsWrap, sidebarEmptyHint, documentTarget: document, el, iconSvg, sidebarRenderSignature, sidebarSessionEntries,
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
      hintModeController: transcript.hintModeController,
      transcriptSlotRuntime: transcript.transcriptSlotRuntime, typingRowRuntime: transcript.typingRowRuntime,
      transcriptScrollRuntime: transcript.transcriptScrollRuntime, transcriptDomRuntime: transcript.transcriptDomRuntime,
      transcriptEventRuntime: transcript.transcriptEventRuntime, transcriptView: transcript.transcriptView,
      markClickLoad: transcript.markClickLoad,
      olderLoadRuntime: historyController.olderLoadRuntime,
      resetChatRenderState: transcript.resetChatRenderState, clearOlderLoadError: historyController.clearOlderLoadError,
      updateChatNavButtons: transcript.updateChatNavButtons, closeChatSearch: transcript.closeChatSearch,
      clearRenderedTranscriptRange: historyController.clearRenderedTranscriptRange, initPageLimit: transcript.initPageLimit,
      dropPendingUserRows: transcript.dropPendingUserRows, updateSessionTranscriptSlot: transcript.updateSessionTranscriptSlot,
      tailCacheMatchesSession: transcript.tailCacheMatchesSession, applySessionListTranscriptIdentity: transcript.applySessionListTranscriptIdentity,
      updateTypingStatsFromSession: transcript.updateTypingStatsFromSession,
      messagePollDelayMs: sendLifecycleController.messagePollDelayMs,
      kickPoll: sendLifecycleController.kickPoll, setPollFastUntilMs: sendLifecycleController.setPollFastUntilMs,
      openMessageEventSource: sendLifecycleController.openMessageEventSource, isMobile, useDesktopSessionActions,
      useTouchFileEditorControls, setSidebarOpen, setSidebarCollapsed, clearCommitUnknownSend, refreshSessions,
      loadOlderMessages: historyController.loadOlderMessages, applySessionRuntimeFromTail: historyController.applySessionRuntimeFromTail,
      renderSessionTail: historyController.renderSessionTail, recoveryDetailsText: historyController.recoveryDetailsText,
      renderPendingTranscriptSlot: historyController.renderPendingTranscriptSlot,
      renderTranscriptLoading: historyController.renderTranscriptLoading, renderTranscriptLoadError: historyController.renderTranscriptLoadError,
      applyCachedTail: historyController.applyCachedTail, jumpToLatest: historyController.jumpToLatest,
      rememberPendingHashSession: historyController.rememberPendingHashSession,
      maybeSelectPendingHashSession: historyController.maybeSelectPendingHashSession,
      dispose: () => {
        attachmentsController.dispose();
        transcript.dispose();
      },
    });
  }

export { createChatInteractionController };
