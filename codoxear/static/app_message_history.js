/* Older-message paging, cursor state, and transcript lifecycle rendering. */
(function installCodoxearMessageHistory(global) {
  "use strict";
  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`message history dependency missing: ${name}`);
    return value;
  }
  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`message history dependency missing: ${name}`);
    return value;
  }
  function createMessageHistoryController(options = {}) {
    const getSelected = requireFunction(options.getSelected, "getSelected");
    const getPollGeneration = requireFunction(options.getPollGeneration, "getPollGeneration");
    const getSessionIndex = requireFunction(options.getSessionIndex, "getSessionIndex");
    const getSessionLifecycleController = requireFunction(options.getSessionLifecycleController, "getSessionLifecycleController");
    const getSessionRefreshController = requireFunction(options.getSessionRefreshController, "getSessionRefreshController");
    const getSendLifecycleController = requireFunction(options.getSendLifecycleController, "getSendLifecycleController");
    const getAttachmentsController = requireFunction(options.getAttachmentsController, "getAttachmentsController");
    const transcript = requireObject(options.transcript, "transcript");
    const { wiring, olderWrap, olderBtn, olderError, olderErrorText, AbortController, performance,
      OLDER_AUTO_COOLDOWN_MS, OLDER_PAGE_LIMIT, api, handleAppAuthLoss, setTurnOpen, setStatus,
      setContext, getCurrentRunning, syncQueueSubmitState, syncComposerSendButton, updateUnattendedBtnState,
      updateQueueBadge, sessionLaunchFailed, confirmApp, setToast, codoxearDisplay, redactedLaunchErrorText,
      chatInner, setTimeout, clearTimeout, sessionIdFromHash, sessionSelectable, isAppDisposed } = options;
    const { transcriptSlotRuntime, transcriptScrollRuntime, clearTranscriptDom, setOlderState,
      restorePendingUserRowsForSession, markClickFirstPaint, updateSessionTranscriptSlot,
      syncActiveTranscriptSlot, updateTypingStatsFromSession, setTyping,
      renderDetachedTranscriptWindow, clearOlderLoadError, showOlderLoadError,
      renderedMessageRows, resetChatRenderState } = transcript;
    const transcriptView = () => transcript.transcriptView();
    const refreshSessions = () => getSessionRefreshController().refreshSessions();
    const kickPoll = (delay = 0) => getSendLifecycleController().kickPoll(delay);
    let pendingHashSessionId = "";
    let pendingHashSessionSelectInFlight = false;
    const olderLoadRuntime = window.CodoxearTranscript.createOlderLoadRuntime(wiring.createOlderLoadOptions({
      olderWrap, olderButton: olderBtn, olderError, olderErrorText, AbortControllerCtor: AbortController,
      nowMs: () => performance.now(), autoCooldownMs: OLDER_AUTO_COOLDOWN_MS,
    }));
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
  return transcript.prependOlderEvents(allEvents, { preserveViewport });
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
  getAttachmentsController().syncAttachButtonState();
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
  return getSendLifecycleController().messageFlowController.applyLiveMessageData(sid, gen, data);
}

async function pollMessages(sid = getSelected(), gen = getPollGeneration()) {
  return getSendLifecycleController().messageFlowController.pollMessages(sid, gen);
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
    return Object.freeze({
      olderLoadRuntime, invalidateOlderLoad, clearOlderLoadError, showOlderLoadError, setOlderState,
      loadTranscriptWindowAtCursor, loadOlderMessages, maybeAutoLoadOlder, clearRenderedTranscriptRange,
      applySessionRuntimeFromTail, renderSessionTail, recoveryDetailsText, syncRecoveryUiForSession,
      renderPendingTranscriptSlot, renderTranscriptLoading, renderTranscriptLoadError, applyCachedTail,
      applyLiveMessageData, pollMessages, jumpToLatest, rememberPendingHashSession, maybeSelectPendingHashSession,
    });
  }
  global.CodoxearMessageHistory = Object.freeze({ createMessageHistoryController });
})(window);
