import * as CodoxearChatNavigation from "./app_chat_navigation.js";
import * as CodoxearChatSearch from "./app_chat_search.js";
import * as CodoxearHintMode from "./app_hint_mode.js";
import * as CodoxearMessageIdentity from "./app_transcript.js";
import * as CodoxearMessageRows from "./app_message_rows.js";
import * as CodoxearTranscript from "./app_transcript.js";
import * as CodoxearTranscriptView from "./app_transcript_view.js";


/* Transcript rendering, viewport state, search wiring, and row projection. */

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`transcript render dependency missing: ${name}`);
    return value;
  }

  function createNavigationPulseController(options = {}) {
    function requireNavigationPulseFunction(value, name) {
      if (typeof value !== "function") throw new TypeError(`navigation pulse controller dependency missing: ${name}`);
      return value;
    }

    const setActiveRow = requireNavigationPulseFunction(options.setActiveRow, "setActiveRow");
    const activeElementIsCopyButton = requireNavigationPulseFunction(options.activeElementIsCopyButton, "activeElementIsCopyButton");
    const setTimeout = requireNavigationPulseFunction(options.setTimeout, "setTimeout");

    function pulseNavigatedRow(row) {
      if (!row) return;
      setActiveRow(row, { focusCopy: activeElementIsCopyButton() });
      row.classList.remove("nav-pulse");
      void row.offsetWidth;
      row.classList.add("nav-pulse");
      setTimeout(() => row.classList.remove("nav-pulse"), 1400);
    }

    return Object.freeze({ pulseNavigatedRow });
  }

  function createTranscriptRenderController(options = {}) {
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
    const getSessionEditController = requireFunction(options.getSessionEditController, "getSessionEditController");
    const getQueueController = requireFunction(options.getQueueController, "getQueueController");
    const isFileViewerOpen = requireFunction(options.isFileViewerOpen, "isFileViewerOpen");
    const upgradeCandidateFileRefs = requireFunction(options.upgradeCandidateFileRefs, "upgradeCandidateFileRefs");
    const isMobile = requireFunction(options.isMobile, "isMobile");
    const refreshSessions = requireFunction(options.refreshSessions, "refreshSessions");
    const jumpToLatest = requireFunction(options.jumpToLatest, "jumpToLatest");
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
      helpViewer, iconSvg, imgInput, isLikelyHeic, isModalTargetOpen,
      isTextEntryElement, jumpBtn, looksLikeImage, modalIsolationTargets, navigator, networkStatus,
      newSessionDialogController, nextUserBtn, olderBtn, olderError, olderErrorText, olderWrap,
      performance, prevUserBtn, pushPerfSample, queueViewer, redactedLaunchErrorText,
      refreshQueueViewer, requestAnimationFrame, resizeComposer, resolveAppUrl,
      safeAttachmentStem, sendChoice, sessionAgentBackend, sessionDisplayName,
      sessionHasOrphanQueueRecovery, sessionHasUnknownSend, sessionIdFromHash, sessionIsFast,
      sessionIsOrphanRecovery, sessionLaunchFailed, sessionLaunchIcon, sessionLaunchLabel,
      sessionLaunchPending, sessionProviderChoice, sessionSelectable, sessionTitleWithId, sessionsWrap,
      setTimeout, setToast, sidebarEffortCode, sidebarEmptyHint, sidebarModelText, sidebarRenderSignature,
      sidebarSessionEntries, storageRemoveItem, storageSetItem, syncComposerSendButton,
      syncQueueSubmitState, textarea, titleLabel, updateUnattendedBtnState,
      window, wiring
    } = options;
    let pendingHashSessionId = "";
    let pendingHashSessionSelectInFlight = false;
    let clickLoadT0 = 0;
    let clickMetricPending = false;
    let attachmentsController = null;
    let messageFlowController = null;
    let currentQueueLen = 0;
    const getHistoryController = requireFunction(options.getHistoryController, "getHistoryController");
    const getSendLifecycleController = requireFunction(options.getSendLifecycleController, "getSendLifecycleController");
    const getAttachmentsController = requireFunction(options.getAttachmentsController, "getAttachmentsController");
function invalidateOlderLoad() {
  getHistoryController().olderLoadRuntime.invalidate();
}

function resetChatRenderState() {
  invalidateOlderLoad();
  transcriptScrollRuntime.enableAutoScroll();
  setSending(false);
  transcriptEventRuntime.resetRecentEvents();
  transcriptSlotRuntime.clearLiveCursor();
  transcriptScrollRuntime.markLiveTail();
  getHistoryController().olderLoadRuntime.resetAutoTrigger();
      clickMetricPending = false;
  transcriptView().replaceWith([]);
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
  // Kept as a private compatibility helper for reset paths. DOM replacement
  // remains owned by TranscriptViewController.
  transcriptView().replaceWith([]);
}

function clearOlderLoadError() {
  getHistoryController().olderLoadRuntime.clearError();
}

function showOlderLoadError() {
  getHistoryController().olderLoadRuntime.showError();
}

function setOlderState({ hasMore, isLoading }) {
  getHistoryController().olderLoadRuntime.setState({ hasMore, isLoading });
}

const codoxearMessageRows = CodoxearMessageRows;
const codoxearTranscriptView = CodoxearTranscriptView;
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
  return CodoxearHintMode.createHintModeController(wiring.createHintModeOptions({
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
  return CodoxearChatNavigation.createChatNavigationController(wiring.createChatNavigationOptions({
    prevUserBtn,
    nextUserBtn,
    getSelected: () => getSelected(),
    getPollGen: () => getPollGeneration(),
    api,
    loadTranscriptWindowAtCursor: (...args) => getHistoryController().loadTranscriptWindowAtCursor(...args),
    loadOlderMessages: (...args) => getHistoryController().loadOlderMessages(...args),
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


chatSearchController = (function instantiateChatSearchController() {
  return CodoxearChatSearch.createChatSearchController(wiring.createChatSearchOptions({
    chatSearchBtn,
    chatSearchInput,
    chatSearchPrevBtn,
    chatSearchNextBtn,
    chatSearchCloseBtn,
    chatSearchStatus,
    chatSearchAllHintEl,
    chatSearchBar,
    createLoadedChatSearchRuntime: CodoxearTranscript.createLoadedChatSearchRuntime,
    createChatSearchAllRuntime: CodoxearTranscript.createChatSearchAllRuntime,
    getSelected: () => getSelected(),
    getPollGen: () => getPollGeneration(),
    api,
    loadTranscriptWindowAtCursor: (...args) => getHistoryController().loadTranscriptWindowAtCursor(...args),
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

const transcriptSlotRuntime = CodoxearTranscript.createTranscriptSlotRuntime(wiring.createTranscriptSlotOptions({
  getSession: (sessionId) => getSessionIndex().get(sessionId) || null,
  maxTailEvents: INIT_PAGE_LIMIT,
}));

function activeTranscriptSnapshot() {
  return transcriptSlotRuntime.activeSnapshot();
}

function initPageLimit() {
  return INIT_PAGE_LIMIT;
}

const typingRowRuntime = CodoxearTranscript.createTypingRowRuntime(wiring.createTypingRowOptions({
  root: chatInner,
  bottomSentinel,
  el,
  shouldAutoScroll: () => transcriptScrollRuntime.snapshot().autoScroll,
  scheduleScrollToBottom: () => transcriptScrollRuntime.scheduleScrollToBottom(),
}));

const transcriptScrollRuntime = CodoxearTranscript.createTranscriptScrollRuntime(wiring.createTranscriptScrollOptions({
  chat,
  jumpButton: jumpBtn,
  timeChip: chatTimeChip,
  requestAnimationFrame: (callback) => requestAnimationFrame(callback),
  hasSelection: () => Boolean(getSelected()),
  isSearchOpen: () => chatSearchController.isOpen(),
  firstVisibleMessageRow,
  dayLabel,
  time24,
  shouldCancelOlderLoad: () => getHistoryController().olderLoadRuntime.shouldCancelOnScroll(),
  cancelOlderLoad: invalidateOlderLoad,
  autoLoadOlder: () => { void getHistoryController().loadOlderMessages({ auto: true }); },
  bottomThresholdPx: 80,
  olderTopTriggerPx: OLDER_TOP_TRIGGER_PX,
  olderCancelPx: OLDER_CANCEL_PX,
}));

const transcriptDomRuntime = CodoxearTranscript.createTranscriptDomRuntime(wiring.createTranscriptDomOptions({
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
  return getHistoryController().olderLoadRuntime.snapshot();
}

function hasOlderMessages() {
  return olderLoadSnapshot().hasMore;
}

function isLoadingOlderMessages() {
  return olderLoadSnapshot().isLoading;
}

function normalizeTailEvent(ev) {
  return CodoxearTranscript.normalizeTailEvent(ev);
}

function normalizeTranscriptState(data) {
  return CodoxearTranscript.normalizeTranscriptState(data);
}

function transcriptKey(threadId, logPath) {
  return CodoxearTranscript.transcriptKey(threadId, logPath);
}

function transcriptSnapshotFromData(data) {
  return CodoxearTranscript.transcriptSnapshotFromData(data);
}

function transcriptIdentityFromData(data, fallback = null) {
  return CodoxearTranscript.transcriptIdentityFromData(data, fallback);
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
  const currentSlot = getSessionTranscriptSlot(sessionId);
  const listedSlot = transcriptSnapshotFromData(sessionMeta);
  const requiresReplacement =
    currentSlot.state === "bound" &&
    (listedSlot.state === "pending_bind" || (listedSlot.state === "bound" && currentSlot.key !== listedSlot.key));
  if (requiresReplacement) {
    // The sidebar is allowed to discover a replacement transcript, but it is
    // not proof that a readable tail is ready. Keep the visible transcript
    // until the guarded same-session reload obtains replacement events.
    void getSessionLifecycleController().openSession(sessionId, { useCache: false, fallbackToCacheOnFailure: true });
    return;
  }

  const slotChange = updateSessionTranscriptSlot(sessionId, sessionMeta);
  if (!slotChange.resetPending) return;

  // NEVER clear the DOM directly from a session-list refresh. This runs
  // after every send (refreshSessions callback) and a race with concurrent
  // polls can set the slot to pending_bind before this fires, reaching this
  // path instead of requiresReplacement above. Delegate to openSession,
  // which fetches replacement content first. For same-session reloads,
  // openSession preserves the existing DOM (messages stay visible).
  void getSessionLifecycleController().openSession(sessionId, { useCache: false, fallbackToCacheOnFailure: true });

  const running = Boolean(sessionMeta.busy);
  const queueLen = Number.isFinite(Number(sessionMeta.queue_len)) ? Number(sessionMeta.queue_len) : 0;
  setTurnOpen(running);
  setStatus({ running, queueLen });
  setContext(sessionMeta.token || null);
  setTyping(running);
}

function updateQueueBadge() {
  const queueController = getQueueController();
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
  return getSendLifecycleController().messageFlowController.updateTypingStatsFromSession(session);
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


      function normalizeTextForPendingMatch(s) {
return CodoxearMessageIdentity.normalizeTextForPendingMatch(s);
      }

      const transcriptEventRuntime = CodoxearTranscript.createTranscriptEventRuntime(wiring.createTranscriptEventOptions({
eventKey: CodoxearMessageIdentity.eventKey,
pendingMatchKey: CodoxearMessageIdentity.pendingMatchKey,
normalizePendingText: CodoxearMessageIdentity.normalizeTextForPendingMatch,
assistantDedupeKey: CodoxearMessageIdentity.chatAssistantDedupeKey,
maxRecentEventKeys: 320,
      }));

      function eventKey(ev) {
return CodoxearMessageIdentity.eventKey(ev);
      }

function markEventSeen(ev) {
  transcriptEventRuntime.markEventSeen(ev);
}

function isDuplicateEvent(ev) {
  return transcriptEventRuntime.isDuplicateEvent(ev);
}

function chatAssistantDedupeKey(ev) {
  return CodoxearMessageIdentity.chatAssistantDedupeKey(ev);
}

function isAdjacentAssistantDuplicateEvent(ev) {
  return transcriptEventRuntime.isAdjacentAssistantDuplicateEvent(ev, {
    renderedAtLiveTail: transcriptScrollRuntime.snapshot().renderedAtLiveTail,
    rows: renderedMessageRows(),
  });
}

function pendingMatchKey(s) {
  return CodoxearMessageIdentity.pendingMatchKey(s);
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
  transcript: CodoxearTranscript,
  getSelectedSessionId: () => getSelected(),
  getMessageRowDeps: messageRowDeps,
  policyRuntime: {
    domRuntime: transcriptDomRuntime,
    scrollRuntime: transcriptScrollRuntime,
    setOlderState,
    getScrollTop: () => chat.scrollTop,
  },
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
function appendEvent(ev) {
  return transcriptView().appendEvents([ev]);
}

function normalizedTranscriptEvents(events, { consumePending = false } = {}) {
  return CodoxearTranscript.normalizedTranscriptEvents(events, {
    consumePending,
    selectedSessionId: getSelected(),
    eventKey,
    takePendingMatch: takePendingUserMatch,
  });
}

function renderTranscript(events, { preserveScroll = false } = {}) {
  return transcriptView().replaceWith(events, { preserveScroll });
}

function renderDetachedTranscriptWindow(events, { hasMore = false, historyCursor = null } = {}) {
  return transcriptView().replaceWith(events, { detached: true, cursor: historyCursor, nextHasMore: hasMore });
}

function prependOlderEvents(events, { preserveViewport = false, historyCursor = null, hasMore = false } = {}) {
  return transcriptView().prependEvents(events, { preserveViewport, cursor: historyCursor, nextHasMore: hasMore });
}

    return Object.freeze({
      transcriptSlotRuntime, typingRowRuntime, transcriptScrollRuntime, transcriptDomRuntime,
      transcriptEventRuntime, chatSearchController, chatNavigationController,
      transcriptView,
      resetChatRenderState, clearOlderLoadError, showOlderLoadError, setOlderState,
      clearTranscriptDom, clearRenderedTranscriptRange: () => getHistoryController().clearRenderedTranscriptRange(),
      initPageLimit, activeTranscriptSnapshot, updateSessionTranscriptSlot, getSessionTranscriptSlot,
      beginTranscriptRenewal, tailCacheMatchesSession, rememberTailSnapshot, appendTailSnapshotEvents,
      applySessionListTranscriptIdentity, updateTypingStatsFromSession, setTyping,
      appendEvent, normalizedTranscriptEvents, renderTranscript, renderDetachedTranscriptWindow,
      prependOlderEvents, dropPendingUserRows, restorePendingUserRowsForSession,
      renderedMessageRows, loadedUserMessageRows, loadedCopyMessageRows, loadedUserJumpTarget,
      loadedCopyJumpTarget, rowSearchText, firstVisibleMessageRow, prefersReducedMotion,
      updateChatNavButtons, closeChatSearch, openChatSearch, clearChatSearchMarks,
      hintModeController,
      eventKey, markEventSeen, isDuplicateEvent, isAdjacentAssistantDuplicateEvent,
      takePendingUserMatch, isTranscriptRenewalCommand,
      transcriptView, markClickFirstPaint, syncActiveTranscriptSlot,
    });
  }

export { createNavigationPulseController, createTranscriptRenderController };
