import * as CodoxearTranscript from "./app_transcript.js";


/* Older-message paging, cursor state, and transcript lifecycle rendering. */
  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`message history dependency missing: ${name}`);
    return value;
  }
  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`message history dependency missing: ${name}`);
    return value;
  }
  function createMessageHistoryController(options = {}) {
    const pollingRuntime = options.pollingRuntime;
    if (!pollingRuntime || typeof pollingRuntime.currentGeneration !== "function") {
      throw new TypeError("message history dependency missing: pollingRuntime");
    }
    const sessionCatalog = options.sessionCatalog;
    if (!sessionCatalog || typeof sessionCatalog.get !== "function") throw new TypeError("message history dependency missing: sessionCatalog");
    const getSessionIndex = () => sessionCatalog.get("sessionIndex");
    const getSessionLifecycleController = requireFunction(options.getSessionLifecycleController, "getSessionLifecycleController");
    const getSessionRefreshController = requireFunction(options.getSessionRefreshController, "getSessionRefreshController");
    const getSendLifecycleController = requireFunction(options.getSendLifecycleController, "getSendLifecycleController");
    const getAttachmentsController = requireFunction(options.getAttachmentsController, "getAttachmentsController");
    const transcript = requireObject(options.transcript, "transcript");
    const sessionState = requireObject(options.sessionState, "sessionState");
    if (typeof sessionState.get !== "function" || typeof sessionState.set !== "function" || typeof sessionState.applyRuntime !== "function") {
      throw new TypeError("message history dependency missing: sessionState");
    }
    const { wiring, olderWrap, olderBtn, olderError, olderErrorText, AbortController, performance,
      OLDER_AUTO_COOLDOWN_MS, OLDER_PAGE_LIMIT, api, handleAppAuthLoss,
      syncQueueSubmitState, syncComposerSendButton, updateUnattendedBtnState,
      sessionLaunchFailed, confirmApp, setToast, codoxearDisplay, redactedLaunchErrorText,
      sessionIdFromHash, sessionSelectable } = options;
    const { transcriptSlotRuntime, transcriptScrollRuntime, setOlderState,
      restorePendingUserRowsForSession, markClickFirstPaint, updateSessionTranscriptSlot,
      syncActiveTranscriptSlot, rememberTailSnapshot, updateTypingStatsFromSession,
      clearOlderLoadError, showOlderLoadError } = transcript;
    const transcriptView = () => transcript.transcriptView();
    const refreshSessions = () => getSessionRefreshController().refreshSessions();
    const kickPoll = (delay = 0) => getSendLifecycleController().kickPoll(delay);
    let pendingHashSessionId = "";
    let pendingHashSessionSelectInFlight = false;
    const olderLoadRuntime = CodoxearTranscript.createOlderLoadRuntime(wiring.createOlderLoadOptions({
      olderWrap, olderButton: olderBtn, olderError, olderErrorText, AbortControllerCtor: AbortController,
      nowMs: () => performance.now(), autoCooldownMs: OLDER_AUTO_COOLDOWN_MS,
    }));
    function invalidateOlderLoad() {
      olderLoadRuntime.invalidate();
    }

    function usableOlderHistoryCursor(data) {
      return CodoxearTranscript.hasUsableOlderHistory(data) ? CodoxearTranscript.historyCursorFromPayload(data) : null;
    }

    function oldestRenderedHistoryCursor() {
      return transcriptView().oldestRenderedHistoryCursor() || transcriptView().historyCursor();
    }

    function clearRenderedTranscriptRange() {
      transcriptView().replaceWith([]);
    }

    function initPageLimit() {
      return INIT_PAGE_LIMIT;
    }

    function olderPageLimit() {
      return OLDER_PAGE_LIMIT;
    }


async function loadTranscriptWindowAtCursor(cursor) {
  const cleanCursor = String(cursor || "").trim();
  if (!sessionState.get("selected") || !cleanCursor) return null;
  const sid = sessionState.get("selected");
  const gen = pollingRuntime.currentGeneration();
  invalidateOlderLoad();
  try {
    const data = await api(`/api/sessions/${sid}/messages/window?cursor=${encodeURIComponent(cleanCursor)}&before=30&after=30`);
    if (sessionState.get("selected") !== sid || pollingRuntime.currentGeneration() !== gen) return null;
    const events = Array.isArray(data.events) ? data.events : [];
    const nextCursor = usableOlderHistoryCursor(data);
    transcriptView().replaceWith(events, { detached: true, cursor: nextCursor, nextHasMore: Boolean(nextCursor) });
    return data;
  } catch (error) {
    if (error && error.status === 401) handleAppAuthLoss();
    else if (sessionState.get("selected") === sid && pollingRuntime.currentGeneration() === gen) showOlderLoadError();
    return null;
  }
}

function prependOlderEvents(allEvents, { preserveViewport = false, historyCursor = null, hasMore = false } = {}) {
  return transcriptView().prependEvents(allEvents, { preserveViewport, cursor: historyCursor, nextHasMore: hasMore });
}

async function loadOlderMessages({ auto = false, cancelOnScroll = true, forcePreserveViewport = null } = {}) {
  const state = olderLoadRuntime.snapshot();
  if (!sessionState.get("selected") || !state.hasMore || state.isLoading) return false;
  if (auto && !olderLoadRuntime.markAutoTrigger()) return false;
  const sid = sessionState.get("selected");
  const gen = pollingRuntime.currentGeneration();
  const load = olderLoadRuntime.beginLoad({ cancelOnScroll });
  const view = typeof transcriptView === "function" ? transcriptView() : null;
  if (view && !view.beginOlderLoad()) {
    olderLoadRuntime.finishLoad(load);
    return false;
  }
  try {
    const reqCursor = oldestRenderedHistoryCursor();
    if (!reqCursor) throw new Error("history cursor missing");
    const data = await api(`/api/sessions/${sid}/messages/history?cursor=${encodeURIComponent(reqCursor)}&limit=${olderPageLimit()}`, {
      signal: load.signal,
    });
    if (sessionState.get("selected") !== sid || pollingRuntime.currentGeneration() !== gen || !olderLoadRuntime.isCurrent(load)) return false;
    const evs = Array.isArray(data.events) ? data.events : [];
    const nextCursor = usableOlderHistoryCursor(data);
    const nextHasOlder = Boolean(nextCursor);
    clearOlderLoadError();
    if (evs.length) {
      if (view) view.prependEvents(evs, { preserveViewport: forcePreserveViewport !== null ? forcePreserveViewport : auto, cursor: nextCursor, nextHasMore: nextHasOlder });
      else {
        prependOlderEvents(evs, { preserveViewport: forcePreserveViewport !== null ? forcePreserveViewport : auto, historyCursor: nextCursor, hasMore: nextHasOlder });
        setOlderState({ hasMore: nextHasOlder, isLoading: false });
      }
      return true;
    }
    if (view) {
      view.olderLoadFailed();
      view.setHistory({ cursor: nextCursor, nextHasMore: nextHasOlder });
    } else {
      setOlderState({ hasMore: nextHasOlder, isLoading: false });
    }
    return false;
  } catch (e) {
    if (e && e.status === 401) {
      handleAppAuthLoss();
      return false;
    }
    if (sessionState.get("selected") !== sid || pollingRuntime.currentGeneration() !== gen || !olderLoadRuntime.isCurrent(load)) return false;
    if (e && e.status === 409) {
      await getSessionLifecycleController().openSession(sid, { useCache: false });
      return false;
    }
    if (view) view.olderLoadFailed();
    else setOlderState({ hasMore: olderLoadRuntime.snapshot().hasMore, isLoading: false });
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
  // The scroll runtime detects the top edge; this controller owns the policy
  // transition into LOADING_OLDER.
  return transcriptView().state().state === "BROWSING";
}

function applySessionRuntimeFromTail(sessionId, data) {
  const slot = syncActiveTranscriptSlot(sessionId);
  transcriptSlotRuntime.setLiveCursor(slot.state === "bound" && typeof data.live_cursor === "string" && data.live_cursor ? data.live_cursor : null);
  // Only update the older-history cursor when the payload actually carries
  // older-history metadata. Poll/SSE responses omit it; overwriting with null
  // would hide the "load older messages" affordance and break scrolling up.
  const incomingCursor = usableOlderHistoryCursor(data);
  const payloadHasOlderInfo = data && (typeof data.has_older !== "undefined" || typeof data.history_cursor !== "undefined");
  if (payloadHasOlderInfo) {
    transcriptView().setHistory({ cursor: incomingCursor, nextHasMore: Boolean(incomingCursor) });
  }
  const nowBusy = Boolean(data && data.busy);
  sessionState.set("turnOpen", nowBusy);
  const queueLen = data && Number.isFinite(Number(data.queue_len)) ? Number(data.queue_len) : 0;
  const session = getSessionIndex().get(sessionId);
  updateTypingStatsFromSession(session, { updateSubagents: false });
  const subagentsRunning = session ? Math.max(0, Math.floor(Number(session.subagents_running) || 0)) : 0;
  sessionState.applyRuntime({ running: nowBusy, queueLen, token: data ? data.token || null : null, subagentsRunning });
  if (slot.state === "bound") {
    const s = getSessionIndex().get(sessionId);
    if (s) rememberTailSnapshot(sessionId, s, data);
  } else {
    transcriptSlotRuntime.deleteTailCache(sessionId);
  }
}

function renderSessionTail(events, { historyCursor = null, hasMore = false } = {}) {
  transcriptView().replaceWith(events, { cursor: historyCursor, nextHasMore: hasMore });
  markClickFirstPaint();
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
  if (sessionState.get("selected") !== sessionId) return;
  const s = getSessionIndex().get(sessionId) || null;
  if (s) {
    const queueLen = Number.isFinite(Number(s.queue_len)) ? Number(s.queue_len) : 0;
    sessionState.applyRuntime({ queueLen });
  }
  getAttachmentsController().syncAttachButtonState();
  syncQueueSubmitState();
  syncComposerSendButton();
  updateUnattendedBtnState();
}

function renderPendingTranscriptSlot(sessionId) {
  transcriptView().replaceWith([]);
  restorePendingUserRowsForSession(sessionId);
  markClickFirstPaint();
}

function renderTranscriptLoading(sessionId) {
  transcriptView().replaceWithLoading();
  restorePendingUserRowsForSession(sessionId);
  markClickFirstPaint();
}

function renderTranscriptLoadError(sessionId, err, { preserveTranscript = false } = {}) {
  if (!preserveTranscript) {
    transcriptView().replaceWith([]);
    restorePendingUserRowsForSession(sessionId);
  }
  const reason = err && err.message ? ` ${err.message}` : "";
  transcriptView().showLoadError({
    message: `Could not load transcript.${reason}`,
    onRetry: (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (sessionState.get("selected") !== sessionId) return;
      void getSessionLifecycleController().openSession(sessionId, { useCache: true });
    },
  });
  sessionState.set("turnOpen", false);
  sessionState.applyRuntime({ running: false });
  markClickFirstPaint();
}

function applyCachedTail(sessionId, cache, sessionMeta) {
  updateSessionTranscriptSlot(sessionId, {
    transcript_state: "bound",
    thread_id: cache.threadId || (sessionMeta ? sessionMeta.thread_id : null),
    log_path: cache.logPath || (sessionMeta ? sessionMeta.log_path : null),
  });
  syncActiveTranscriptSlot(sessionId);
  transcriptSlotRuntime.setLiveCursor(cache.liveCursor || null);
  transcriptView().setHistory({
    cursor: typeof cache.historyCursor === "string" && cache.historyCursor ? cache.historyCursor : null,
    nextHasMore: Boolean(cache.hasOlder),
  });
  renderSessionTail(cache.events, {
    historyCursor: typeof cache.historyCursor === "string" && cache.historyCursor ? cache.historyCursor : null,
    hasMore: Boolean(cache.hasOlder),
  });
  const metaBusy = Boolean(sessionMeta && sessionMeta.busy);
  const cachedBusy = Boolean(cache.busy) || metaBusy;
  const queueLen =
    sessionMeta && Number.isFinite(Number(sessionMeta.queue_len))
      ? Number(sessionMeta.queue_len)
      : Number.isFinite(Number(cache.queueLen))
        ? Number(cache.queueLen)
        : 0;
  sessionState.set("turnOpen", cachedBusy);
  updateTypingStatsFromSession(sessionMeta, { updateSubagents: false });
  const subagentsRunning = sessionMeta ? Math.max(0, Math.floor(Number(sessionMeta.subagents_running) || 0)) : 0;
  sessionState.applyRuntime({
    running: cachedBusy,
    queueLen,
    token: cache.token || (sessionMeta ? sessionMeta.token || null : null),
    subagentsRunning,
  });
}

async function applyLiveMessageData(sid, gen, data) {
  return getSendLifecycleController().messageFlowController.applyLiveMessageData(sid, gen, data);
}

async function pollMessages(sid = sessionState.get("selected"), gen = pollingRuntime.currentGeneration()) {
  return getSendLifecycleController().messageFlowController.pollMessages(sid, gen);
}

async function jumpToLatest() {
  if (!sessionState.get("selected")) return;
  const sid = sessionState.get("selected");
  invalidateOlderLoad();
  const view = typeof transcriptView === "function" ? transcriptView() : null;
  if (view && typeof view.scrollToBottom === "function") view.scrollToBottom({ force: true });
  else transcriptScrollRuntime.enableAutoScroll();
  try {
    await getSessionLifecycleController().openSession(sid, {
      useCache: false,
      fallbackToCacheOnFailure: true,
      forceRender: true,
    });
  } catch (e) {
    if (sessionState.get("selected") !== sid) return;
    setToast(`jump error: ${e && e.message ? e.message : "unknown error"}`);
  }
  if (sessionState.get("selected") !== sid) return;
  if (view && typeof view.scrollToBottom === "function") view.scrollToBottom();
  else transcriptScrollRuntime.scheduleScrollToBottom({ syncJump: true });
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
  if (sid === sessionState.get("selected")) {
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

export { createMessageHistoryController };
