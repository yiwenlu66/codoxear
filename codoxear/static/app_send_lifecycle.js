
/* Confirmed send, staged attachments, queue coordination, and live delivery. */
  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`send lifecycle dependency missing: ${name}`);
    return value;
  }
  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`send lifecycle dependency missing: ${name}`);
    return value;
  }
  function createSendLifecycleController(options = {}) {
    const currentGeneration = options.currentGeneration;
    if (typeof currentGeneration !== "function") {
      throw new TypeError("send lifecycle dependency missing: currentGeneration");
    }
    const sessionCatalog = options.sessionCatalog;
    if (!sessionCatalog || typeof sessionCatalog.get !== "function") throw new TypeError("send lifecycle dependency missing: sessionCatalog");
    const getSessionIndex = () => sessionCatalog.get("sessionIndex");
    const getSessionLifecycleController = requireFunction(options.getSessionLifecycleController, "getSessionLifecycleController");
    const getSessionRefreshController = requireFunction(options.getSessionRefreshController, "getSessionRefreshController");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.applyRuntime !== "function") {
      throw new TypeError("send lifecycle dependency missing: sessionState");
    }
    const isAppDisposed = requireFunction(options.isAppDisposed, "isAppDisposed");
    const getSessionEditController = requireFunction(options.getSessionEditController, "getSessionEditController");
    const getQueueController = requireFunction(options.getQueueController, "getQueueController");
    const isFileViewerOpen = requireFunction(options.isFileViewerOpen, "isFileViewerOpen");
    const upgradeCandidateFileRefs = requireFunction(options.upgradeCandidateFileRefs, "upgradeCandidateFileRefs");
    const {
      ATTACH_UPLOAD_MAX_BYTES, AbortController, EventSource, $, addAppEvent, api, attachBtn, b64FromBytes, chatInner, codoxearAttachments, codoxearMessageFlow, composer,
      confirmApp, dataTransferHasFiles, document, el,
      extractFilesFromClipboardData, extractFilesFromDropData, fmtBytes, handleAppAuthLoss,
      imgInput, isLikelyHeic, looksLikeImage, navigator, networkStatus,
      resizeComposer, resolveAppUrl,
      safeAttachmentStem, sessionHasOrphanQueueRecovery, sessionHasUnknownSend, sessionIsOrphanRecovery, sessionLaunchFailed, setTimeout, setToast, syncComposerSendButton,
      syncQueueSubmitState, textarea, window, wiring
    } = options;
    const transcript = requireObject(options.transcript, "transcript");
    const history = requireObject(options.history, "history");
    let attachmentsController = null;
    let messageFlowController = null;
    const refreshSessions = () => getSessionRefreshController().refreshSessions();
    const {
      activeTranscriptSnapshot, updateSessionTranscriptSlot, resetChatRenderState,
      getSessionTranscriptSlot, beginTranscriptRenewal,
      dropPendingUserRows, appendTailSnapshotEvents, initPageLimit, typingRowRuntime,
      transcriptSlotRuntime, transcriptEventRuntime, transcriptScrollRuntime, isTranscriptRenewalCommand,
      transcriptView, updateTypingStatsFromSession
    } = transcript;
    const { renderPendingTranscriptSlot, renderSessionTail, applySessionRuntimeFromTail,
      syncRecoveryUiForSession } = history;
attachmentsController = codoxearAttachments.createAttachmentsController(wiring.createAttachmentsOptions({
  sessionState,
  attachBtn,
  imgInput,
  composer,
  textarea,
  sessionCatalog,
  sessionLaunchFailed,
  sessionHasUnknownSend,
  sessionIsOrphanRecovery,
  sessionHasOrphanQueueRecovery,
  api,
  setToast,
  handleAppAuthLoss,
  refreshSessions,
  setPollFastUntilMs,
  kickPoll,
  resizeComposer,
  getTray: () => $("#stagedAttachments"),
  el,
  fmtBytes,
  safeAttachmentStem,
  isLikelyHeic,
  looksLikeImage,
  b64FromBytes,
  dataTransferHasFiles,
  extractFilesFromClipboardData,
  extractFilesFromDropData,
  addEventListener: addAppEvent,
  uploadMaxBytes: ATTACH_UPLOAD_MAX_BYTES,
}));

messageFlowController = codoxearMessageFlow.createMessageFlowController(wiring.createMessageFlowOptions({
  currentGeneration,
  isAppDisposed: () => isAppDisposed(),
  sessionCatalog,
  sessionLaunchFailed,
  api,
  resolveAppUrl,
  handleAppAuthLoss,
  refreshSessions,
  openSession: (...args) => getSessionLifecycleController().openSession(...args),
  clearSelectedSessionAfterRemoval: (...args) => getSessionLifecycleController().clearSelectedSessionAfterRemoval(...args),
  activeTranscriptSnapshot,
  updateSessionTranscriptSlot,
  renderPendingTranscriptSlot,
  renderSessionTail,
  applySessionRuntimeFromTail,
  resetChatRenderState,
  setAttachCount: (count) => attachmentsController.setAttachCount(count),
  setLiveCursor: (cursor) => transcriptSlotRuntime.setLiveCursor(cursor),
  appendEvents: (events) => transcriptView().appendEvents(events),
  appendTailSnapshotEvents,
  sessionState,
  initPageLimit,
  typingRowRuntime,
  getStagedAttachments: () => attachmentsController.getStagedAttachments(),
  normalizedStagedAttachments: (list) => attachmentsController.normalizedStagedAttachments(list),
  setSelectedSessionPendingAttachment: (sessionId, value) => attachmentsController.setSelectedSessionPendingAttachment(sessionId, value),
  syncSendButtonState: syncComposerSendButton,
  syncAttachButtonState: () => attachmentsController.syncAttachButtonState(),
  syncQueueSubmitState,
  syncRecoveryUiForSession,
  confirmAction: (options) => confirmApp(options),
  setToast,
  isTranscriptRenewalCommand,
  nextLocalEchoId: () => transcriptEventRuntime.nextLocalEchoId(),
  renderedAtLiveTail: () => transcriptView().state().renderedAtLiveTail,
  getSessionTranscriptSlot,
  addPendingUser: (pending) => transcriptEventRuntime.addPendingUser(pending),
  deleteTailCache: (sessionId) => transcriptSlotRuntime.deleteTailCache(sessionId),
  beginTranscriptRenewal,
  clearLiveCursor: () => transcriptSlotRuntime.clearLiveCursor(),
  invalidateOlderLoad: () => history.invalidateOlderLoad(),
  dropPendingUser: (sessionId, localId) => transcriptEventRuntime.dropPendingUsers(sessionId, (pending) => pending && pending.id === localId),
  removePendingUserRow: (localId) => {
    const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${localId}"]`);
    if (!pendingEl) return;
    const pendingRow = pendingEl.closest(".msg-row");
    if (pendingRow) pendingRow.remove();
    else pendingEl.remove();
  },
  hasPendingForSession: (sessionId) => transcriptEventRuntime.hasPendingForSession(sessionId),
  visibilityState: () => document.visibilityState,
  navigatorValue: () => (typeof navigator === "undefined" ? undefined : navigator),
  reportTransportSuccess: () => networkStatus.reportSuccess(),
  reportTransportFailure: () => networkStatus.reportFailure(),
  EventSource: typeof EventSource === "function" ? EventSource : null,
  AbortController: typeof AbortController === "function" ? AbortController : null,
  setTimeout: window.setTimeout.bind(window),
  clearTimeout: window.clearTimeout.bind(window),
  now: () => Date.now(),
  consoleWarn: (...args) => console.warn(...args),
  consoleError: (...args) => console.error(...args),
}));

function messagePollDelayMs(now = Date.now()) {
  return messageFlowController.messagePollDelayMs(now);
}

function kickPoll(ms = 0) {
  return messageFlowController.kickPoll(ms);
}

function setPollFastUntilMs(value) {
  messageFlowController.setPollFastUntilMs(value);
}

function openMessageEventSource(sessionId = sessionState.get("selected"), generation = currentGeneration()) {
  return messageFlowController.openMessageEventSource(sessionId, generation);
}

    return Object.freeze({
      attachmentsController, messageFlowController, messagePollDelayMs, kickPoll, setPollFastUntilMs,
      openMessageEventSource,
    });
  }

export { createSendLifecycleController };
