/* Chat interaction composition: transcript rendering, history navigation, search, status, and live delivery assembly. */
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
    const setSending = requireFunction(options.setSending, "setSending");
    const getTurnOpen = requireFunction(options.getTurnOpen, "getTurnOpen");
    const setTurnOpen = requireFunction(options.setTurnOpen, "setTurnOpen");
    const getCurrentRunning = requireFunction(options.getCurrentRunning, "getCurrentRunning");
    const setCurrentRunning = requireFunction(options.setCurrentRunning, "setCurrentRunning");
    const getCurrentSubagentsRunning = requireFunction(options.getCurrentSubagentsRunning, "getCurrentSubagentsRunning");
    const setCurrentSubagentsRunning = requireFunction(options.setCurrentSubagentsRunning, "setCurrentSubagentsRunning");
    const isAppDisposed = requireFunction(options.isAppDisposed, "isAppDisposed");
    const { setStatus, setContext,
      $, ATTACH_UPLOAD_MAX_BYTES, AbortController, CHAT_DOM_WINDOW, CHAT_DOM_WINDOW_WITH_HISTORY_SLACK,
      EventSource, INIT_PAGE_LIMIT, Node, OLDER_AUTO_COOLDOWN_MS, OLDER_CANCEL_PX, OLDER_PAGE_LIMIT,
      OLDER_TOP_TRIGGER_PX, addAppEvent, agentBackendDisplayName, agentBackendLogoPath, api, appConfirm,
      attachBtn, b64FromBytes, baseName, bottomSentinel, chat, chatInner, chatMarkdownHtmlCached,
      chatSearchAllHintEl, chatSearchBar, chatSearchBtn, chatSearchCloseBtn, chatSearchInput,
      chatSearchNextBtn, chatSearchPrevBtn, chatSearchStatus, chatTimeChip, codeBlockCopyRuntime,
      codoxearAttachments, codoxearCodeCopy, codoxearDisplay, codoxearMessageFlow, codoxearModal,
      codoxearNavigationPulse, codoxearPendingUser, codoxearSessions, codoxearViewport, composer,
      confirmApp, copyToClipboard, dataTransferHasFiles, diagViewer, document, editViewer, el,
      extractFilesFromClipboardData, extractFilesFromDropData, fmtBytes, fmtRelativeAge, handleAppAuthLoss,
      helpViewer, iconSvg, imgInput, isFileViewerOpen, isLikelyHeic, isModalTargetOpen,
      isTextEntryElement, jumpBtn, looksLikeImage, modalIsolationTargets, navigator, networkStatus,
      newSessionDialogController, nextUserBtn, olderBtn, olderError, olderErrorText, olderWrap,
      performance, prevUserBtn, pushPerfSample, queueController, queueViewer, redactedLaunchErrorText,
      refreshQueueViewer, renderStatusChip, requestAnimationFrame, resizeComposer, resolveAppUrl,
      safeAttachmentStem, sendChoice, sessionAgentBackend, sessionDisplayName, sessionEditController,
      sessionHasOrphanQueueRecovery, sessionHasUnknownSend, sessionIdFromHash, sessionIsFast,
      sessionIsOrphanRecovery, sessionLaunchFailed, sessionLaunchIcon, sessionLaunchLabel,
      sessionLaunchPending, sessionProviderChoice, sessionSelectable, sessionTitleWithId, sessionsWrap,
      setTimeout, setToast, sidebarEffortCode, sidebarEmptyHint, sidebarModelText, sidebarRenderSignature,
      sidebarSessionEntries, storageRemoveItem, storageSetItem, syncComposerSendButton,
      syncQueueSubmitState, textarea, titleLabel, updateUnattendedBtnState, upgradeCandidateFileRefs,
      window, wiring
    } = options;
    let pendingHashSessionId = "";
    let pendingHashSessionSelectInFlight = false;
    let clickLoadT0 = 0;
    let clickMetricPending = false;
    let attachmentsController = null;
    let messageFlowController = null;
    let currentQueueLen = 0;
    let lastToken = null;
function invalidateOlderLoad() {
  olderLoadRuntime.invalidate();
}

function resetChatRenderState() {
  invalidateOlderLoad();
  transcriptScrollRuntime.enableAutoScroll();
  setSending(false);
  transcriptEventRuntime.resetRecentEvents();
  transcriptSlotRuntime.clearLiveCursor();
  transcriptScrollRuntime.markLiveTail();
  olderLoadRuntime.resetAutoTrigger();
      clickMetricPending = false;
  clearTranscriptDom();
  messageCopyNavigationRuntime.reset();
      setOlderState({ hasMore: false, isLoading: false });
  typingRowRuntime.reset();
  jumpBtn.style.display = "none";
      updateChatNavButtons();
      if (chatSearchController.isOpen()) closeChatSearch();
  transcriptScrollRuntime.reset({ scrollTop: 0 });
      transcriptScrollRuntime.syncVisibleTimeIndicator();
	        }

function clearTranscriptDom() {
  transcriptDomRuntime.clear();
}

function clearOlderLoadError() {
  olderLoadRuntime.clearError();
}

function showOlderLoadError() {
  olderLoadRuntime.showError();
}

function setOlderState({ hasMore, isLoading }) {
  olderLoadRuntime.setState({ hasMore, isLoading });
}

const codoxearMessageRows = window.CodoxearMessageRows;
const codoxearTranscriptView = window.CodoxearTranscriptView;
if (
  !codoxearMessageRows ||
  !codoxearTranscriptView ||
  typeof codoxearTranscriptView.createTranscriptViewController !== "function" ||
  typeof codoxearMessageRows.makeRow !== "function" ||
  typeof codoxearMessageRows.safeMakeRow !== "function" ||
  typeof codoxearMessageRows.messageCopyButtonForRow !== "function" ||
  typeof codoxearMessageRows.renderedMessageRows !== "function" ||
  typeof codoxearMessageRows.loadedUserMessageRows !== "function" ||
  typeof codoxearMessageRows.loadedCopyMessageRows !== "function" ||
  typeof codoxearMessageRows.activeElementIsMessageCopyButton !== "function" ||
  typeof codoxearMessageRows.createMessageCopyNavigationRuntime !== "function" ||
  typeof codoxearMessageRows.rowSearchText !== "function" ||
  typeof codoxearMessageRows.compareRowsInDomOrder !== "function" ||
  typeof codoxearMessageRows.loadedUserJumpTarget !== "function" ||
  typeof codoxearMessageRows.loadedCopyJumpTarget !== "function" ||
  typeof codoxearMessageRows.clearChatSearchMarks !== "function" ||
  typeof codoxearMessageRows.applyChatSearchMarks !== "function" ||
  typeof codoxearMessageRows.oldestRenderedHistoryCursor !== "function" ||
  typeof codoxearMessageRows.firstVisibleMessageRow !== "function" ||
  typeof codoxearMessageRows.trimRenderedRowTargets !== "function" ||
  typeof codoxearMessageRows.trimRowsBeforeViewportTargets !== "function"
)
  throw new Error("Codoxear transcript view helpers failed to load");

let transcriptViewController = null;

function transcriptView() {
  if (!transcriptViewController) throw new Error("transcript view controller is not initialized");
  return transcriptViewController;
}

const messageCopyNavigationRuntime = codoxearMessageRows.createMessageCopyNavigationRuntime(wiring.createMessageCopyNavigationOptions({ root: chatInner }));

function renderedMessageRows() {
  return transcriptView().renderedMessageRows();
}

function loadedUserMessageRows() {
  return transcriptView().loadedUserMessageRows();
}

function loadedCopyMessageRows() {
  return transcriptView().loadedCopyMessageRows();
}

function messageCopyButtonForRow(row) {
  return codoxearMessageRows.messageCopyButtonForRow(row);
}

function activeElementIsMessageCopyButton() {
  return codoxearMessageRows.activeElementIsMessageCopyButton(document);
}

function rowSearchText(row) {
  return codoxearMessageRows.rowSearchText(row);
}

function compareRowsInDomOrder(a, b) {
  return codoxearMessageRows.compareRowsInDomOrder(a, b, Node);
}

function loadedUserJumpTarget(rows, direction, threshold) {
  return codoxearMessageRows.loadedUserJumpTarget(rows, direction, threshold);
}

function loadedCopyJumpTarget(rows, direction, threshold) {
  return messageCopyNavigationRuntime.jumpTarget(rows, direction, threshold);
}

function applyChatSearchMarks(matches, currentRow, query) {
  return transcriptView().applyChatSearchMarks(matches, currentRow, query);
}

function firstVisibleMessageRow() {
  return codoxearMessageRows.firstVisibleMessageRow(renderedMessageRows(), chat.scrollTop + 1);
}

function syncMessageCopyTabStops() {
  messageCopyNavigationRuntime.syncTabStops(renderedMessageRows());
}

function setActiveMessageCopyRow(row, { focusCopy = false } = {}) {
  messageCopyNavigationRuntime.setActiveRow(row, { focusCopy });
}

addAppEvent(chatInner, "pointerover", (e) => {
  if (activeElementIsMessageCopyButton()) return;
  const row = e.target && typeof e.target.closest === "function" ? e.target.closest(".msg-row") : null;
  if (row && chatInner.contains(row)) setActiveMessageCopyRow(row);
});

addAppEvent(chatInner, "focusin", (e) => {
  const row = e.target && typeof e.target.closest === "function" ? e.target.closest(".msg-row") : null;
  if (row && chatInner.contains(row)) setActiveMessageCopyRow(row);
});

function isTouchCopyMode() {
  return window.matchMedia("(max-width: 700px), (pointer: coarse)").matches;
}

addAppEvent(chatInner, "click", (e) => {
  if (!isTouchCopyMode() || window.getSelection().toString()) return;
  const target = e.target && typeof e.target.closest === "function" ? e.target : null;
  if (!target || target.closest("a, button, input, select, textarea, [role='link'], mark")) return;
  const pre = codoxearCodeCopy.codePreFromTarget(target);
  if (pre && chatInner.contains(pre)) {
    codeBlockCopyRuntime.toggleTouchPre(pre, chatInner);
    return;
  }
  const bubble = target.closest(".msg");
  const row = bubble && bubble.closest(".msg-row");
  if (!row || !chatInner.contains(row) || (target !== bubble && !target.closest(".md"))) return;
  messageCopyNavigationRuntime.toggleTouchRow(row);
});

function prefersReducedMotion() {
  return codoxearViewport.prefersReducedMotion();
}

const navigationPulseController = codoxearNavigationPulse.createNavigationPulseController(wiring.createNavigationPulseOptions({
  setActiveRow: setActiveMessageCopyRow,
  activeElementIsCopyButton: activeElementIsMessageCopyButton,
  setTimeout,
}));

const hintModeController = (function instantiateHintModeController() {
  const codoxearHintMode = window.CodoxearHintMode;
  if (!codoxearHintMode || typeof codoxearHintMode.createHintModeController !== "function")
    throw new Error("Codoxear hint mode controller failed to load");
  return codoxearHintMode.createHintModeController(wiring.createHintModeOptions({
    documentTarget: document,
    isTextEntryElement,
    isMobile,
    modalIsolationTargets,
    isModalTargetOpen,
    addAppEvent,
    shellHints: Array.from(document.querySelectorAll("[data-hint]")).map((element) => ({
      label: element.getAttribute("data-hint"),
      element,
    })),
  }));
})();

const activateModalButtonForKey = codoxearModal.createModalKeyboardHandler(wiring.createModalKeyboardHandlerOptions({
  modalIsolationTargets,
  isTextEntryElement,
}));
addAppEvent(document, "keydown", activateModalButtonForKey);

// --- Direct (no-leader) Vimium-style shortcuts ---
// These fire when not in a text-entry element, no modal is open, and
// hint mode is not active. They don't conflict with hint-mode letters
// (which require `f` leader first) — the context disambiguates.
addAppEvent(document, "keydown", (e) => {
  if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.isComposing) return;
  if (hintModeController && hintModeController.isActive()) return;
  if (isTextEntryElement(document.activeElement)) return;
  if (isModalTargetOpen(appConfirm) || isModalTargetOpen(sendChoice) || isModalTargetOpen(queueViewer) || isModalTargetOpen(helpViewer) || isModalTargetOpen(diagViewer) || isModalTargetOpen(editViewer) || newSessionDialogController.isOpen() || isFileViewerOpen()) return;
  // Capital keys (Shift) are distinct actions: G = go to bottom, D = delete.
  if (e.shiftKey) {
    const shifted = String(e.key || "");
    if (shifted === "G") {
      e.preventDefault();
      void jumpToLatest();
      return;
    }
    if (shifted === "D") {
      e.preventDefault();
      void (async () => {
        if (!getSelected()) return;
        const sid = getSelected();
        const confirmed = await confirmApp({
          title: "Delete session?",
          message: "Delete the current session? This cannot be undone.",
          confirmText: "Delete",
          cancelText: "Cancel",
          destructive: true,
        });
        if (!confirmed) return;
        if (getSelected() !== sid) return;
        try {
          await api(`/api/sessions/${sid}/delete`, { method: "POST", body: {} });
          getSessionLifecycleController().clearDeletedSessionClientState(sid);
          await refreshSessions();
          setToast("session deleted");
        } catch (err) {
          setToast(`delete error: ${err && err.message ? err.message : "unknown error"}`);
        }
      })();
      return;
    }
    return;
  }
  const key = String(e.key || "").toLowerCase();
  if (key === "i") {
    e.preventDefault();
    textarea.focus({ preventScroll: true });
    return;
  }
  const step = Math.round(chat.clientHeight * 0.15);
  const halfPage = Math.round(chat.clientHeight * 0.45);
  const scrollBehavior = prefersReducedMotion() ? "auto" : "smooth";
  if (key === "j") {
    e.preventDefault();
    chat.scrollBy({ top: step, behavior: scrollBehavior });
  } else if (key === "k") {
    e.preventDefault();
    chat.scrollBy({ top: -step, behavior: scrollBehavior });
  } else if (key === "d") {
    e.preventDefault();
    chat.scrollBy({ top: halfPage, behavior: scrollBehavior });
  } else if (key === "u") {
    e.preventDefault();
    chat.scrollBy({ top: -halfPage, behavior: scrollBehavior });
  }
});

// Loaded-chat navigation rail + direct-to-search shortcut orchestration
// lives in the CodoxearChatNavigation controller
// (codoxear/static/app_chat_navigation.js). app.js keeps DOM
// construction for prevUserBtn/nextUserBtn and the thin wrappers below.
// The controller is instantiated after the DOM nodes and message-row
// helpers exist; it wires the prev/next button handlers and the
// document keydown listener itself. Chat search internals stay here.
const chatNavigationController = (function instantiateChatNavigationController() {
  const codoxearChatNavigation = window.CodoxearChatNavigation;
  if (!codoxearChatNavigation || typeof codoxearChatNavigation.createChatNavigationController !== "function")
    throw new Error("Codoxear chat navigation controller failed to load");
  return codoxearChatNavigation.createChatNavigationController(wiring.createChatNavigationOptions({
    prevUserBtn,
    nextUserBtn,
    getSelected: () => getSelected(),
    getPollGen: () => getPollGeneration(),
    api,
    loadTranscriptWindowAtCursor,
    loadOlderMessages,
    loadedUserMessageRows,
    loadedCopyMessageRows,
    loadedUserJumpTarget,
    loadedCopyJumpTarget,
    getScrollTop: () => chat.scrollTop,
    prefersReducedMotion,
    pulseNavigatedRow: (row) => navigationPulseController.pulseNavigatedRow(row),
    setToast,
    openChatSearch,
    handleAppAuthLoss,
    isTextEntryElement,
    modalIsolationTargets,
    isModalTargetOpen,
    addAppEvent,
    documentTarget: document,
  }));
})();

function updateChatNavButtons() {
  chatNavigationController.syncButtons();
}

function jumpToLoadedUserMessage(direction) {
  chatNavigationController.jumpToLoadedUserMessage(direction);
}

function jumpToLoadedMessage(direction) {
  chatNavigationController.jumpToLoadedMessage(direction);
}

function clearChatSearchMarks() {
  transcriptView().clearChatSearchMarks();
}

function compactChatSearchSnippet(text, query, limit = 96) {
  return codoxearDisplay.compactChatSearchSnippet(text, query, limit);
}

function chatSearchTranscriptHint(match, query) {
  return codoxearDisplay.chatSearchTranscriptHint(match, query);
}

// --- Loaded-chat search + older-history search orchestration now
// lives in the CodoxearChatSearch controller
// (codoxear/static/app_chat_search.js). app.js keeps DOM construction
// for the search bar/controls, the row/text/mark helpers, transcript
// rendering + older-load authority, and the thin wrappers below that
// other app.js call sites and the chat navigation controller use.
let chatSearchController;

function openChatSearch() {
  chatSearchController.open();
}

function closeChatSearch() {
  chatSearchController.close();
}

function refreshLoadedChatSearch(options) {
  chatSearchController.refreshLoaded(options);
}

function stepChatSearch(delta) {
  return chatSearchController.step(delta);
}

// The direct-to-search shortcut (`/`) lives in the
// CodoxearChatNavigation controller (codoxear/static/app_chat_navigation.js),
// wired via chatNavigationController above.

let activeTailHistoryCursor = null;

function usableOlderHistoryCursor(data) {
  return codoxearTranscript.hasUsableOlderHistory(data) ? codoxearTranscript.historyCursorFromPayload(data) : null;
}

function oldestRenderedHistoryCursor() {
  return transcriptView().oldestRenderedHistoryCursor() || activeTailHistoryCursor;
}

function clearRenderedTranscriptRange() {
  activeTailHistoryCursor = null;
  setOlderState({ hasMore: false, isLoading: false });
  transcriptScrollRuntime.markLiveTail();
}

function initPageLimit() {
  return INIT_PAGE_LIMIT;
}

function olderPageLimit() {
  return OLDER_PAGE_LIMIT;
}

const codoxearTranscript = window.CodoxearTranscript;
if (
  !codoxearTranscript ||
  typeof codoxearTranscript.normalizeTailEvent !== "function" ||
  typeof codoxearTranscript.normalizeTranscriptState !== "function" ||
  typeof codoxearTranscript.normalizedTranscriptEvents !== "function" ||
  typeof codoxearTranscript.transcriptKey !== "function" ||
  typeof codoxearTranscript.historyCursorFromPayload !== "function" ||
  typeof codoxearTranscript.hasUsableOlderHistory !== "function" ||
  typeof codoxearTranscript.transcriptSnapshotFromData !== "function" ||
  typeof codoxearTranscript.transcriptIdentityFromData !== "function" ||
  typeof codoxearTranscript.tailCacheMatchesSession !== "function" ||
  typeof codoxearTranscript.rememberTailSnapshot !== "function" ||
  typeof codoxearTranscript.appendTailSnapshotEvents !== "function" ||
  typeof codoxearTranscript.createTranscriptSlotRuntime !== "function" ||
  typeof codoxearTranscript.createTypingRowRuntime !== "function" ||
  typeof codoxearTranscript.hasHumanOriginatedUserEvent !== "function" ||
  typeof codoxearTranscript.createTranscriptRenderRuntime !== "function" ||
  typeof codoxearTranscript.createTranscriptDomRuntime !== "function" ||
  typeof codoxearTranscript.createTranscriptScrollRuntime !== "function" ||
  typeof codoxearTranscript.createTranscriptEventRuntime !== "function" ||
  typeof codoxearTranscript.createOlderLoadRuntime !== "function" ||
  typeof codoxearTranscript.createLoadedChatSearchRuntime !== "function" ||
  typeof codoxearTranscript.createChatSearchAllRuntime !== "function"
)
  throw new Error("Codoxear transcript helpers failed to load");

const olderLoadRuntime = codoxearTranscript.createOlderLoadRuntime(wiring.createOlderLoadOptions({
  olderWrap,
  olderButton: olderBtn,
  olderError,
  olderErrorText,
  AbortControllerCtor: AbortController,
  nowMs: () => performance.now(),
  autoCooldownMs: OLDER_AUTO_COOLDOWN_MS,
}));

chatSearchController = (function instantiateChatSearchController() {
  const codoxearChatSearch = window.CodoxearChatSearch;
  if (!codoxearChatSearch || typeof codoxearChatSearch.createChatSearchController !== "function")
    throw new Error("Codoxear chat search controller failed to load");
  return codoxearChatSearch.createChatSearchController(wiring.createChatSearchOptions({
    chatSearchBtn,
    chatSearchInput,
    chatSearchPrevBtn,
    chatSearchNextBtn,
    chatSearchCloseBtn,
    chatSearchStatus,
    chatSearchAllHintEl,
    chatSearchBar,
    createLoadedChatSearchRuntime: codoxearTranscript.createLoadedChatSearchRuntime,
    createChatSearchAllRuntime: codoxearTranscript.createChatSearchAllRuntime,
    getSelected: () => getSelected(),
    getPollGen: () => getPollGeneration(),
    api,
    loadTranscriptWindowAtCursor,
    handleAppAuthLoss,
    syncVisibleTimeIndicator: () => transcriptScrollRuntime.syncVisibleTimeIndicator(),
    renderedMessageRows,
    rowSearchText,
    clearChatSearchMarks,
    applyChatSearchMarks,
    pulseNavigatedRow: (row) => navigationPulseController.pulseNavigatedRow(row),
    prefersReducedMotion,
  }));
})();

const transcriptSlotRuntime = codoxearTranscript.createTranscriptSlotRuntime(wiring.createTranscriptSlotOptions({
  getSession: (sessionId) => getSessionIndex().get(sessionId) || null,
  maxTailEvents: INIT_PAGE_LIMIT,
}));

function activeTranscriptSnapshot() {
  return transcriptSlotRuntime.activeSnapshot();
}

const typingRowRuntime = codoxearTranscript.createTypingRowRuntime(wiring.createTypingRowOptions({
  root: chatInner,
  bottomSentinel,
  el,
  shouldAutoScroll: () => transcriptScrollRuntime.snapshot().autoScroll,
  scheduleScrollToBottom: () => transcriptScrollRuntime.scheduleScrollToBottom(),
}));

const transcriptScrollRuntime = codoxearTranscript.createTranscriptScrollRuntime(wiring.createTranscriptScrollOptions({
  chat,
  jumpButton: jumpBtn,
  timeChip: chatTimeChip,
  requestAnimationFrame: (callback) => requestAnimationFrame(callback),
  hasSelection: () => Boolean(getSelected()),
  isSearchOpen: () => chatSearchController.isOpen(),
  firstVisibleMessageRow,
  dayLabel,
  time24,
  shouldCancelOlderLoad: () => olderLoadRuntime.shouldCancelOnScroll(),
  cancelOlderLoad: invalidateOlderLoad,
  autoLoadOlder: () => { void loadOlderMessages({ auto: true }); },
  bottomThresholdPx: 80,
  olderTopTriggerPx: OLDER_TOP_TRIGGER_PX,
  olderCancelPx: OLDER_CANCEL_PX,
}));

const transcriptDomRuntime = codoxearTranscript.createTranscriptDomRuntime(wiring.createTranscriptDomOptions({
  root: chatInner,
  olderWrap,
  bottomSentinel,
  el,
  ymd,
  dayLabel,
  getRenderedRows: renderedMessageRows,
  trimRenderedRowTargets: codoxearMessageRows.trimRenderedRowTargets,
  trimRowsBeforeViewportTargets: codoxearMessageRows.trimRowsBeforeViewportTargets,
  scrollRuntime: transcriptScrollRuntime,
  defaultWindowRows: CHAT_DOM_WINDOW,
  afterDecorate: () => {
    updateChatNavButtons();
    syncMessageCopyTabStops();
    if (chatSearchController.isOpen()) chatSearchController.refreshLoaded({ jump: false, preserveCurrent: true });
  },
}));

function olderLoadSnapshot() {
  return olderLoadRuntime.snapshot();
}

function hasOlderMessages() {
  return olderLoadSnapshot().hasMore;
}

function isLoadingOlderMessages() {
  return olderLoadSnapshot().isLoading;
}

function normalizeTailEvent(ev) {
  return codoxearTranscript.normalizeTailEvent(ev);
}

function normalizeTranscriptState(data) {
  return codoxearTranscript.normalizeTranscriptState(data);
}

function transcriptKey(threadId, logPath) {
  return codoxearTranscript.transcriptKey(threadId, logPath);
}

function transcriptSnapshotFromData(data) {
  return codoxearTranscript.transcriptSnapshotFromData(data);
}

function transcriptIdentityFromData(data, fallback = null) {
  return codoxearTranscript.transcriptIdentityFromData(data, fallback);
}

function getSessionTranscriptSlot(sessionId) {
  return transcriptSlotRuntime.getSlot(sessionId);
}

function syncActiveTranscriptSlot(sessionId) {
  return transcriptSlotRuntime.syncActiveSlot(sessionId);
}

function dropPendingUserRows(sessionId, predicate = null) {
  if (!sessionId) return;
  const dropped = transcriptEventRuntime.dropPendingUsers(sessionId, predicate);
  if (getSelected() !== sessionId) return;
  for (const item of dropped) {
    if (!item || !item.id) continue;
    const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${item.id}"]`);
    const row = pendingEl ? pendingEl.closest(".msg-row") : null;
    if (row) row.remove();
  }
}

function updateSessionTranscriptSlot(sessionId, data) {
  const change = transcriptSlotRuntime.updateSlot(sessionId, data);
  if (change.resetPending) dropPendingUserRows(sessionId, () => true);
  if (getSelected() === sessionId) syncActiveTranscriptSlot(sessionId);
  return change;
}

function beginTranscriptRenewal(sessionId) {
  const change = transcriptSlotRuntime.beginRenewal(sessionId);
  if (!change) return;
  dropPendingUserRows(sessionId, () => true);
  if (getSelected() === sessionId) syncActiveTranscriptSlot(sessionId);
}

function tailCacheMatchesSession(cache, session) {
  return transcriptSlotRuntime.tailCacheMatchesSession(cache, session);
}

function rememberTailSnapshot(sessionId, session, data) {
  return transcriptSlotRuntime.rememberTail(sessionId, session, data);
}

function appendTailSnapshotEvents(sessionId, events, { session = null, identityData = null, liveCursor: nextLiveCursor, busy, queueLen, token } = {}) {
  return transcriptSlotRuntime.appendTailEvents(sessionId, events, {
    session,
    identityData,
    liveCursor: nextLiveCursor,
    busy,
    queueLen,
    token,
  });
}

function restorePendingUserRowsForSession(sessionId) {
  if (!sessionId) return;
  const slot = getSessionTranscriptSlot(sessionId);
  const items = transcriptEventRuntime.pendingUsersForSession(sessionId, Number(slot.epoch || 0));
  for (const item of items) {
    if (!item || !item.id) continue;
    if (chatInner.querySelector(`.msg.user[data-local-id="${item.id}"]`)) continue;
    appendEvent({ role: "user", text: item.text, pending: true, localId: item.id, ts: item.t0 });
  }
}

function applySessionListTranscriptIdentity(sessionId, sessionMeta) {
  if (!sessionId || getSelected() !== sessionId || !sessionMeta) return;
  const slotChange = updateSessionTranscriptSlot(sessionId, sessionMeta);
  if (!slotChange.resetPending) return;

  transcriptSlotRuntime.deleteTailCache(sessionId);
  transcriptSlotRuntime.clearLiveCursor();
  clearRenderedTranscriptRange();
  attachmentsController.setAttachCount(0);
  invalidateOlderLoad();
  transcriptEventRuntime.resetRecentEvents();
  transcriptScrollRuntime.enableAutoScroll();
  clearTranscriptDom();
  if (slotChange.current.state === "pending_bind") {
    renderPendingTranscriptSlot(sessionId);
  } else {
    setOlderState({ hasMore: false, isLoading: false });
    transcriptScrollRuntime.syncJumpButton();
    kickPoll(0);
  }

  const running = Boolean(sessionMeta.busy);
  const queueLen = Number.isFinite(Number(sessionMeta.queue_len)) ? Number(sessionMeta.queue_len) : 0;
  setTurnOpen(running);
  setStatus({ running, queueLen });
  setContext(sessionMeta.token || null);
  setTyping(running);
}

function updateQueueBadge() {
  if (queueController) {
    queueController.updateQueueBadge();
    if (queueViewer.style.display === "flex") void refreshQueueViewer();
  }
}

  function markClickFirstPaint() {
    if (!clickMetricPending) return;
    clickMetricPending = false;
    const dt = performance.now() - clickLoadT0;
    pushPerfSample("click_to_first_message_ms", dt);
  }

function updateTypingStatsFromSession(session) {
  return messageFlowController.updateTypingStatsFromSession(session);
}

function setTyping(show) {
  typingRowRuntime.setVisible(show);
  typingRowRuntime.setSubagentVisible(!show);
}

function ymd(d) {
  return codoxearDisplay.ymd(d);
}

function dayLabel(d) {
  return codoxearDisplay.dayLabel(d);
}

function time24(d) {
  return codoxearDisplay.time24(d);
}

function rebuildDecorations({ preserveScroll }) {
  transcriptDomRuntime.rebuildDecorations({ preserveScroll });
}

function trimRenderedRows({ fromTop, maxRows = CHAT_DOM_WINDOW }) {
  transcriptDomRuntime.trimRenderedRows({ fromTop, maxRows });
}

function trimRenderedRowsBeforeViewport({ maxRows = CHAT_DOM_WINDOW } = {}) {
  transcriptDomRuntime.trimRowsBeforeViewport({ maxRows, viewportTop: chat.scrollTop + 1 });
}

function messageRowDeps() {
  return {
    el,
    chatMarkdownHtmlCached,
    upgradeCandidateFileRefs,
    time24,
    iconSvg,
    copyToClipboard,
    setToast,
    chatAssistantDedupeKey,
    setTimeout: window.setTimeout.bind(window),
    consoleError: console.error.bind(console),
  };
}

      const codoxearMessageIdentity = window.CodoxearMessageIdentity;
      if (
!codoxearMessageIdentity ||
typeof codoxearMessageIdentity.normalizeTextForPendingMatch !== "function" ||
typeof codoxearMessageIdentity.pendingMatchKey !== "function" ||
typeof codoxearMessageIdentity.eventKey !== "function" ||
typeof codoxearMessageIdentity.chatAssistantDedupeKey !== "function"
      )
throw new Error("Codoxear message identity helpers failed to load");

      function normalizeTextForPendingMatch(s) {
return codoxearMessageIdentity.normalizeTextForPendingMatch(s);
      }

      const transcriptEventRuntime = codoxearTranscript.createTranscriptEventRuntime(wiring.createTranscriptEventOptions({
eventKey: codoxearMessageIdentity.eventKey,
pendingMatchKey: codoxearMessageIdentity.pendingMatchKey,
normalizePendingText: codoxearMessageIdentity.normalizeTextForPendingMatch,
assistantDedupeKey: codoxearMessageIdentity.chatAssistantDedupeKey,
maxRecentEventKeys: 320,
      }));

      function eventKey(ev) {
return codoxearMessageIdentity.eventKey(ev);
      }

function markEventSeen(ev) {
  transcriptEventRuntime.markEventSeen(ev);
}

function isDuplicateEvent(ev) {
  return transcriptEventRuntime.isDuplicateEvent(ev);
}

function chatAssistantDedupeKey(ev) {
  return codoxearMessageIdentity.chatAssistantDedupeKey(ev);
}

function isAdjacentAssistantDuplicateEvent(ev) {
  return transcriptEventRuntime.isAdjacentAssistantDuplicateEvent(ev, {
    renderedAtLiveTail: transcriptScrollRuntime.snapshot().renderedAtLiveTail,
    rows: renderedMessageRows(),
  });
}

function pendingMatchKey(s) {
  return codoxearMessageIdentity.pendingMatchKey(s);
}

      function isTranscriptRenewalCommand(raw, sessionId = getSelected()) {
const session = sessionId ? getSessionIndex().get(sessionId) : null;
if (!session || sessionAgentBackend(session) !== "codex") return false;
return String(raw || "").trim() === "/new";
      }

      function takePendingUserMatch(ev, sessionId = getSelected(), { allowUntimedCommit = true } = {}) {
const slot = getSessionTranscriptSlot(sessionId);
return transcriptEventRuntime.takePendingUserMatch(ev, sessionId, Number(slot.epoch || 0), { allowUntimedCommit });
      }

      const pendingUserController = codoxearPendingUser.createPendingUserController(wiring.createPendingUserOptions({
selectedSessionId: () => getSelected(),
takePendingUserMatch,
chatInner,
markdownHtml: chatMarkdownHtmlCached,
time24,
rebuildDecorations,
markEventSeen,
      }));

transcriptViewController = codoxearTranscriptView.createTranscriptViewController(wiring.createTranscriptViewOptions({
  root: chatInner,
  bottomSentinel,
  document,
  el,
  messageRows: codoxearMessageRows,
  transcript: codoxearTranscript,
  getSelectedSessionId: () => getSelected(),
  getMessageRowDeps: messageRowDeps,
  renderRuntime: {
    normalizeEvents: normalizedTranscriptEvents,
    consumePendingUserIfMatches: (event, sessionId) => pendingUserController.consumePendingUserIfMatches(event, sessionId),
    isDuplicateEvent,
    isAdjacentAssistantDuplicateEvent,
    markEventSeen,
    markFirstPaint: markClickFirstPaint,
    restorePendingRows: restorePendingUserRowsForSession,
    resetRecentEvents: () => transcriptEventRuntime.resetRecentEvents(),
    setOlderState,
    firstVisibleMessageRow,
    getScrollTop: () => chat.scrollTop,
    getSelectedSessionId: () => getSelected(),
    domRuntime: transcriptDomRuntime,
    scrollRuntime: transcriptScrollRuntime,
    typingRowRuntime,
    historySlackRows: CHAT_DOM_WINDOW_WITH_HISTORY_SLACK,
  },
}));

attachmentsController = codoxearAttachments.createAttachmentsController(wiring.createAttachmentsOptions({
  attachBtn,
  imgInput,
  composer,
  textarea,
  getSelected: () => getSelected(),
  getSessionInfo: (sessionId) => getSessionIndex().get(sessionId) || null,
  patchSessionInfo: (sessionId, patch) => {
    const current = getSessionIndex().get(sessionId);
    if (!current) return;
    Object.assign(current, patch || {});
    getSessionIndex().set(sessionId, current);
  },
  getSending,
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
  getSelected: () => getSelected(),
  getGeneration: () => getPollGeneration(),
  isAppDisposed: () => isAppDisposed(),
  getTurnOpen,
  setTurnOpen,
  getSessionInfo: (sessionId) => getSessionIndex().get(sessionId) || null,
  patchSessionInfo: (sessionId, patch) => {
    const current = getSessionIndex().get(sessionId);
    if (!current) return;
    Object.assign(current, patch || {});
    getSessionIndex().set(sessionId, current);
  },
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
  appendEvent,
  appendTailSnapshotEvents,
  setStatus,
  setContext,
  setTyping,
  setSubagentsRunning: (value) => {
    setCurrentSubagentsRunning(value);
    renderStatusChip();
  },
  updateSessionTitle: (session) => { titleLabel.textContent = sessionTitleWithId(session); },
  initPageLimit,
  typingRowRuntime,
  getSending,
  setSending,
  getCurrentRunning,
  setCurrentRunning,
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
  renderedAtLiveTail: () => transcriptScrollRuntime.snapshot().renderedAtLiveTail,
  clearTranscriptDom,
  clearRenderedTranscriptRange,
  setOlderState,
  getSessionTranscriptSlot,
  addPendingUser: (pending) => transcriptEventRuntime.addPendingUser(pending),
  deleteTailCache: (sessionId) => transcriptSlotRuntime.deleteTailCache(sessionId),
  beginTranscriptRenewal,
  clearLiveCursor: () => transcriptSlotRuntime.clearLiveCursor(),
  invalidateOlderLoad,
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

function openMessageEventSource(sessionId = getSelected(), generation = getPollGeneration()) {
  return messageFlowController.openMessageEventSource(sessionId, generation);
}

function isMobile() {
  return codoxearViewport.isMobile();
}

function useDesktopSessionActions() {
  return codoxearViewport.useDesktopSessionActions();
}

function useTouchFileEditorControls() {
  return codoxearViewport.useTouchFileEditorControls();
}

function setSidebarOpen(open) {
  if (open) {
    document.body.classList.add("sidebar-open");
    storageSetItem("codexweb.sidebarOpen", "1");
  } else {
    document.body.classList.remove("sidebar-open");
    storageRemoveItem("codexweb.sidebarOpen");
  }
}

function setSidebarCollapsed(collapsed) {
  if (collapsed) {
    document.body.classList.add("sidebar-collapsed");
    storageSetItem("codexweb.sidebarCollapsed", "1");
  } else {
    document.body.classList.remove("sidebar-collapsed");
    storageRemoveItem("codexweb.sidebarCollapsed");
  }
}

function clearCommitUnknownSend(sid, previewText = "") {
  return getSessionLifecycleController().clearCommitUnknownSend(sid, previewText);
}


 const sidebarController = codoxearSessions.createSessionsController(wiring.createSessionsOptions({
   sessionsWrap,
   sidebarEmptyHint,
   el,
   iconSvg,
   sidebarRenderSignature,
   sidebarSessionEntries,
   sessionDisplayName,
   sessionLaunchFailed,
   sessionLaunchPending,
   redactedLaunchErrorText,
   fmtRelativeAge,
   sidebarEffortCode,
   sidebarModelText,
   baseName,
   sessionIsFast,
   agentBackendLogoPath,
   agentBackendDisplayName,
   sessionAgentBackend,
   sessionLaunchIcon,
   sessionLaunchLabel,
   confirmAction: (options) => confirmApp(options),
   api,
   clearDeletedSessionClientState: (...args) => getSessionLifecycleController().clearDeletedSessionClientState(...args),
   refreshSessions,
   setToast,
   openEditSession: (sid) => sessionEditController.openEditSession(sid),
   duplicateSession: async (session) => {
     const cwd = session && session.cwd && session.cwd !== "?" ? session.cwd : "";
     if (!cwd) {
       setToast("cwd unavailable");
       return;
     }
     await getSessionLifecycleController().spawnSessionWithCwd(
       cwd,
       null,
       null,
       "",
       sessionProviderChoice(session),
       session && session.model ? session.model : "default",
       session && session.reasoning_effort ? session.reasoning_effort : "high",
       sessionIsFast(session),
       !!(session && session.transport === "tmux"),
       null,
       sessionAgentBackend(session)
     );
   },
   selectSession: (...args) => getSessionLifecycleController().selectSession(...args),
   setSidebarOpen,
   now: () => Date.now(),
   performanceNow: () => performance.now(),
   consoleError: (...args) => console.error(...args),
 }));

function refreshSessions() {
  return getSessionRefreshController().refreshSessions();
}


function appendEvent(ev) {
  transcriptView().appendEvent(ev);
}

function normalizedTranscriptEvents(events, { consumePending = false } = {}) {
  return codoxearTranscript.normalizedTranscriptEvents(events, {
    consumePending,
    selectedSessionId: getSelected(),
    eventKey,
    takePendingMatch: takePendingUserMatch,
  });
}

function renderTranscript(events, { preserveScroll = false } = {}) {
  return transcriptView().renderTranscript(events, { preserveScroll });
}

function renderDetachedTranscriptWindow(events, { hasMore = false } = {}) {
  return transcriptView().renderDetachedTranscriptWindow(events, { hasMore });
}

async function loadTranscriptWindowAtCursor(cursor) {
  const cleanCursor = String(cursor || "").trim();
  if (!getSelected() || !cleanCursor) return null;
  const sid = getSelected();
  const gen = getPollGeneration();
  invalidateOlderLoad();
  try {
    const data = await api(`/api/sessions/${sid}/messages/window?cursor=${encodeURIComponent(cleanCursor)}&before=30&after=30`);
    if (getSelected() !== sid || getPollGeneration() !== gen) return null;
    const events = Array.isArray(data.events) ? data.events : [];
    activeTailHistoryCursor = usableOlderHistoryCursor(data);
    setOlderState({ hasMore: Boolean(activeTailHistoryCursor), isLoading: false });
    if (!renderDetachedTranscriptWindow(events, { hasMore: Boolean(activeTailHistoryCursor) })) return null;
    return data;
  } catch (error) {
    if (error && error.status === 401) handleAppAuthLoss();
    else if (getSelected() === sid && getPollGeneration() === gen) showOlderLoadError();
    return null;
  }
}

function prependOlderEvents(allEvents, { preserveViewport = false } = {}) {
  return transcriptView().prependOlderEvents(allEvents, { preserveViewport });
}

async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {}) {
  const state = olderLoadSnapshot();
  if (!getSelected() || !state.hasMore || state.isLoading) return false;
  if (auto && !olderLoadRuntime.markAutoTrigger()) return false;
  const sid = getSelected();
  const gen = getPollGeneration();
  const load = olderLoadRuntime.beginLoad({ cancelOnScroll });
  try {
    const reqCursor = oldestRenderedHistoryCursor();
    if (!reqCursor) throw new Error("history cursor missing");
    const data = await api(`/api/sessions/${sid}/messages/history?cursor=${encodeURIComponent(reqCursor)}&limit=${olderPageLimit()}`, {
      signal: load.signal,
    });
    if (getSelected() !== sid || getPollGeneration() !== gen || !olderLoadRuntime.isCurrent(load)) return false;
    const evs = Array.isArray(data.events) ? data.events : [];
    activeTailHistoryCursor = usableOlderHistoryCursor(data);
    const nextHasOlder = Boolean(activeTailHistoryCursor);
    clearOlderLoadError();
    setOlderState({ hasMore: nextHasOlder, isLoading: false });
    if (evs.length) {
      prependOlderEvents(evs, { preserveViewport: auto });
      return true;
    }
    return false;
  } catch (e) {
    if (e && e.status === 401) {
      handleAppAuthLoss();
      return false;
    }
    if (getSelected() !== sid || getPollGeneration() !== gen || !olderLoadRuntime.isCurrent(load)) return false;
    if (e && e.status === 409) {
      await getSessionLifecycleController().openSession(sid, { useCache: false });
      return false;
    }
    setOlderState({ hasMore: hasOlderMessages(), isLoading: false });
    showOlderLoadError();
    return false;
  } finally {
    olderLoadRuntime.finishLoad(load);
  }
}

// Older-history search window loading (loadNearestOlderChatSearchWindow /
// loadChatSearchCursorWindow) now lives in the CodoxearChatSearch
// controller (codoxear/static/app_chat_search.js). app.js keeps the
// transcript/older-load authority those paths invoke through injected
// deps (olderLoadRuntime, loadOlderMessages, renderDetachedTranscript
// Window, openSession, handleAppAuthLoss, invalidateOlderLoad,
// setOlderState, showOlderLoadError).

function maybeAutoLoadOlder() {
  transcriptScrollRuntime.maybeAutoLoadOlder();
}

function applySessionRuntimeFromTail(sessionId, data) {
  const slot = syncActiveTranscriptSlot(sessionId);
  transcriptSlotRuntime.setLiveCursor(slot.state === "bound" && typeof data.live_cursor === "string" && data.live_cursor ? data.live_cursor : null);
  activeTailHistoryCursor = usableOlderHistoryCursor(data);
  setOlderState({ hasMore: Boolean(activeTailHistoryCursor), isLoading: false });
  const nowBusy = Boolean(data && data.busy);
  setTurnOpen(nowBusy);
  const queueLen = data && Number.isFinite(Number(data.queue_len)) ? Number(data.queue_len) : 0;
  const session = getSessionIndex().get(sessionId);
  updateTypingStatsFromSession(session);
  setStatus({ running: nowBusy, queueLen });
  setContext(data ? data.token : null);
  setTyping(nowBusy);
  if (slot.state === "bound") {
    const s = getSessionIndex().get(sessionId);
    if (s) rememberTailSnapshot(sessionId, s, data);
  } else {
    transcriptSlotRuntime.deleteTailCache(sessionId);
  }
}

function renderSessionTail(events) {
  renderTranscript(events, { preserveScroll: false });
  markClickFirstPaint();
  transcriptScrollRuntime.scheduleScrollToBottom({ double: true });
}


function recoveryPromptPreview(text, maxLen = 320) {
  return codoxearDisplay.recoveryPromptPreview(text, maxLen);
}

function recoveryDetailsText(sessionId, s) {
  const lines = [
    "Codoxear recovery details",
    `Session: ${sessionId}`,
  ];
  if (s && s.cwd) lines.push(`cwd: ${s.cwd}`);
  if (s && s.agent_backend) lines.push(`backend: ${s.agent_backend}`);
  if (s && sessionLaunchFailed(s)) {
    lines.push("state: launch failed");
    if (s.launch_stage) lines.push(`launch stage: ${s.launch_stage}`);
    const safeLaunchError = redactedLaunchErrorText(s.launch_error);
    if (safeLaunchError) lines.push(`launch error: ${safeLaunchError}`);
    if (s.model_provider) lines.push(`model provider: ${s.model_provider}`);
    if (s.model) lines.push(`model: ${s.model}`);
    if (s.reasoning_effort) lines.push(`reasoning: ${s.reasoning_effort}`);
    if (s.service_tier) lines.push(`service tier: ${s.service_tier}`);
    if (s.tmux_session || s.tmux_window) lines.push(`tmux: ${s.tmux_session || "-"}${s.tmux_window ? ":" + s.tmux_window : ""}`);
    const submitted = Number.isFinite(Number(s.submitted_user_message_count)) ? Number(s.submitted_user_message_count) : 0;
    if (submitted > 0) lines.push(`submitted prompts: ${submitted}`);
  }
  if (s && s.orphan_recovery) lines.push("state: missing session/orphan recovery");
  if (s && s.queue_recovery) lines.push("state: queued recovery items present");
  if (s && s.commit_unknown_send) lines.push("state: direct send commit unknown");
  const qn = s && Number.isFinite(Number(s.queue_len)) ? Number(s.queue_len) : 0;
  if (qn > 0) lines.push(`queued recovery items: ${qn}`);
  const preview = recoveryPromptPreview(s && s.commit_unknown_send_text ? s.commit_unknown_send_text : "", 2000);
  if (preview) lines.push("", "Unknown-send prompt:", preview);
  return lines.join("\n");
}

async function dismissFailedLaunchRecord(sessionId) {
  const s = getSessionIndex().get(sessionId);
  if (!sessionLaunchFailed(s)) {
    setToast("launch record is not failed");
    return;
  }
  const confirmed = await confirmApp({
    title: "Dismiss launch record?",
    message: "Dismiss this launch record?",
    confirmText: "Dismiss",
    cancelText: "Cancel",
    destructive: true,
  });
  if (!confirmed) return;
  try {
    await api(`/api/sessions/${sessionId}/delete`, { method: "POST", body: {} });
    getSessionLifecycleController().clearDeletedSessionClientState(sessionId);
    await refreshSessions();
    setToast("Dismissed launch record");
  } catch (err) {
    setToast(`dismiss error: ${err && err.message ? err.message : "unknown error"}`);
  }
}

function syncRecoveryUiForSession(sessionId) {
  if (getSelected() !== sessionId) return;
  const s = getSessionIndex().get(sessionId) || null;
  if (s) {
    const queueLen = Number.isFinite(Number(s.queue_len)) ? Number(s.queue_len) : 0;
    setStatus({ running: getCurrentRunning(), queueLen });
  }
  attachmentsController.syncAttachButtonState();
  syncQueueSubmitState();
  syncComposerSendButton();
  updateUnattendedBtnState();
  updateQueueBadge();
}

function renderPendingTranscriptSlot(sessionId) {
  clearTranscriptDom();
  activeTailHistoryCursor = null;
  setOlderState({ hasMore: false, isLoading: false });
  transcriptScrollRuntime.markLiveTail();
  restorePendingUserRowsForSession(sessionId);
  markClickFirstPaint();
  transcriptScrollRuntime.syncJumpButton();
}

function renderTranscriptLoading(sessionId) {
  clearTranscriptDom();
  activeTailHistoryCursor = null;
  setOlderState({ hasMore: false, isLoading: false });
  transcriptScrollRuntime.markLiveTail();
  restorePendingUserRowsForSession(sessionId);
  transcriptView().renderLoadingRow();
  transcriptScrollRuntime.syncJumpButton();
}

function renderTranscriptLoadError(sessionId, err, { preserveTranscript = false } = {}) {
  for (const row of Array.from(chatInner.querySelectorAll(".transcript-error-row"))) row.remove();
  if (!preserveTranscript) {
    clearTranscriptDom();
    activeTailHistoryCursor = null;
    setOlderState({ hasMore: false, isLoading: false });
    transcriptScrollRuntime.markLiveTail();
    restorePendingUserRowsForSession(sessionId);
  }
  const reason = err && err.message ? ` ${err.message}` : "";
  transcriptView().renderLoadErrorRow({
    message: `Could not load transcript.${reason}`,
    onRetry: (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (getSelected() !== sessionId) return;
      void getSessionLifecycleController().openSession(sessionId, { useCache: true });
    },
  });
  setTurnOpen(false);
  setTyping(false);
  markClickFirstPaint();
  transcriptScrollRuntime.syncJumpButton();
}

function applyCachedTail(sessionId, cache, sessionMeta) {
  updateSessionTranscriptSlot(sessionId, {
    transcript_state: "bound",
    thread_id: cache.threadId || (sessionMeta ? sessionMeta.thread_id : null),
    log_path: cache.logPath || (sessionMeta ? sessionMeta.log_path : null),
  });
  syncActiveTranscriptSlot(sessionId);
  transcriptSlotRuntime.setLiveCursor(cache.liveCursor || null);
  activeTailHistoryCursor = typeof cache.historyCursor === "string" && cache.historyCursor ? cache.historyCursor : null;
  setOlderState({ hasMore: Boolean(cache.hasOlder && activeTailHistoryCursor), isLoading: false });
  renderSessionTail(cache.events);
  const metaBusy = Boolean(sessionMeta && sessionMeta.busy);
  const cachedBusy = Boolean(cache.busy) || metaBusy;
  const queueLen =
    sessionMeta && Number.isFinite(Number(sessionMeta.queue_len))
      ? Number(sessionMeta.queue_len)
      : Number.isFinite(Number(cache.queueLen))
        ? Number(cache.queueLen)
        : 0;
  setTurnOpen(cachedBusy);
  setStatus({ running: cachedBusy, queueLen });
  setContext(cache.token || (sessionMeta ? sessionMeta.token : null));
  updateTypingStatsFromSession(sessionMeta);
  setTyping(cachedBusy);
}

async function applyLiveMessageData(sid, gen, data) {
  return messageFlowController.applyLiveMessageData(sid, gen, data);
}

async function pollMessages(sid = getSelected(), gen = getPollGeneration()) {
  return messageFlowController.pollMessages(sid, gen);
}

async function jumpToLatest() {
  if (!getSelected()) return;
  const sid = getSelected();
  invalidateOlderLoad();
  transcriptScrollRuntime.enableAutoScroll();
  try {
    await getSessionLifecycleController().openSession(sid, { useCache: false, fallbackToCacheOnFailure: true });
  } catch (e) {
    if (getSelected() !== sid) return;
    setToast(`jump error: ${e && e.message ? e.message : "unknown error"}`);
  }
  if (getSelected() !== sid) return;
  transcriptScrollRuntime.scheduleScrollToBottom({ syncJump: true });
  kickPoll(0);
}

function rememberPendingHashSession(sid) {
  pendingHashSessionId = String(sid || "").trim();
}

function maybeSelectPendingHashSession() {
  const sid = pendingHashSessionId;
  if (!sid || pendingHashSessionSelectInFlight) return;
  if (sessionIdFromHash() !== sid) {
    rememberPendingHashSession("");
    return;
  }
  if (sid === getSelected()) {
    rememberPendingHashSession("");
    return;
  }
  const session = getSessionIndex().get(sid);
  if (!sessionSelectable(session)) return;
  rememberPendingHashSession("");
  pendingHashSessionSelectInFlight = true;
  void getSessionLifecycleController().selectSession(sid)
    .catch((e) => {
      if (e && e.status === 401) handleAppAuthLoss();
      else console.error("pending hash session select failed", e);
    })
    .finally(() => {
      pendingHashSessionSelectInFlight = false;
    });
}

// Unattended menu state, async load/save orchestration, input draft
// handling, menu focus/visibility, and control event handling live in
// the CodoxearUnattended controller (codoxear/static/app_unattended.js).
// app.js owns DOM construction for the unattended button/menu/controls,
// the updateUnattendedBtnState shell projection (which delegates the
// unattended-specific projection to the controller), and the thin
// delegating wrappers below. The controller is instantiated after the
// DOM nodes exist; it wires the button/menu/input handlers and the
// document Escape/click + window resize listeners itself.

    return Object.freeze({
      attachmentsController, messageFlowController, chatSearchController, chatNavigationController,
      transcriptSlotRuntime, typingRowRuntime, transcriptScrollRuntime, transcriptDomRuntime,
      transcriptEventRuntime, olderLoadRuntime,
      resetChatRenderState, clearOlderLoadError, updateChatNavButtons, closeChatSearch,
      clearRenderedTranscriptRange, initPageLimit, dropPendingUserRows, updateSessionTranscriptSlot,
      tailCacheMatchesSession, applySessionListTranscriptIdentity, updateQueueBadge,
      updateTypingStatsFromSession, setTyping, messagePollDelayMs, kickPoll, setPollFastUntilMs,
      openMessageEventSource, isMobile, useDesktopSessionActions, useTouchFileEditorControls,
      setSidebarOpen, setSidebarCollapsed, clearCommitUnknownSend, refreshSessions,
      loadOlderMessages, applySessionRuntimeFromTail, renderSessionTail, recoveryDetailsText,
      syncRecoveryUiForSession, renderPendingTranscriptSlot, renderTranscriptLoading,
      renderTranscriptLoadError, applyCachedTail, jumpToLatest, rememberPendingHashSession,
      maybeSelectPendingHashSession, getSending,
      getCurrentRunning,
    });
  }

  global.CodoxearChatInteraction = Object.freeze({ createChatInteractionController });
})(window);
