(function () {
  "use strict";

  function normalizeTailEvent(ev) {
    if (!ev || (ev.role !== "user" && ev.role !== "assistant")) return null;
    if (typeof ev.text !== "string" || !ev.text.trim()) return null;
    const out = { role: ev.role, text: ev.text };
    if (typeof ev.ts === "number" && Number.isFinite(ev.ts)) out.ts = ev.ts;
    if (typeof ev.message_class === "string") out.message_class = ev.message_class;
    if (typeof ev.message_id === "string") out.message_id = ev.message_id;
    if (typeof ev.notification_text === "string") out.notification_text = ev.notification_text;
    if (typeof ev.history_cursor === "string" && ev.history_cursor) out.history_cursor = ev.history_cursor;
    return out;
  }

  function normalizeTranscriptState(data) {
    const raw = data && typeof data.transcript_state === "string" ? data.transcript_state : "";
    if (raw === "bound" || raw === "pending_bind" || raw === "failed") return raw;
    return data && typeof data.log_path === "string" && data.log_path ? "bound" : "pending_bind";
  }

  function transcriptKey(threadId, logPath) {
    if (typeof threadId !== "string" || !threadId) return null;
    if (typeof logPath !== "string" || !logPath) return null;
    return `${threadId}\n${logPath}`;
  }

  function transcriptSnapshotFromData(data) {
    const state = normalizeTranscriptState(data);
    const threadId = state === "bound" && typeof data?.thread_id === "string" && data.thread_id ? data.thread_id : null;
    const logPath = state === "bound" && typeof data?.log_path === "string" && data.log_path ? data.log_path : null;
    return {
      state,
      threadId,
      logPath,
      key: state === "bound" ? transcriptKey(threadId, logPath) : null,
    };
  }

  function transcriptIdentityFromData(data, fallback = null) {
    const dataThreadId = typeof data?.thread_id === "string" && data.thread_id ? data.thread_id : null;
    const dataLogPath = typeof data?.log_path === "string" && data.log_path ? data.log_path : null;
    const fallbackThreadId =
      fallback && typeof fallback.thread_id === "string" && fallback.thread_id
        ? fallback.thread_id
        : fallback && typeof fallback.threadId === "string" && fallback.threadId
          ? fallback.threadId
          : null;
    const fallbackLogPath =
      fallback && typeof fallback.log_path === "string" && fallback.log_path
        ? fallback.log_path
        : fallback && typeof fallback.logPath === "string" && fallback.logPath
          ? fallback.logPath
          : null;
    return { threadId: dataThreadId || fallbackThreadId, logPath: dataLogPath || fallbackLogPath };
  }

  function tailCacheMatchesSession(cache, session) {
    if (!cache || !session) return false;
    const cacheThreadId = typeof cache.threadId === "string" ? cache.threadId : null;
    const cacheLogPath = typeof cache.logPath === "string" ? cache.logPath : null;
    const sessionThreadId = typeof session.thread_id === "string" ? session.thread_id : null;
    const sessionLogPath = typeof session.log_path === "string" ? session.log_path : null;
    return cacheThreadId === sessionThreadId && cacheLogPath === sessionLogPath;
  }

  function rememberTailSnapshot(tailCache, sessionId, session, data, maxEvents) {
    if (!sessionId || !session || !data || typeof data !== "object") return;
    if (normalizeTranscriptState(data) !== "bound") {
      tailCache.delete(sessionId);
      return;
    }
    const events = [];
    for (const ev of Array.isArray(data.events) ? data.events : []) {
      const norm = normalizeTailEvent(ev);
      if (norm) events.push(norm);
    }
    const limit = Math.max(0, Number(maxEvents) || 0);
    if (limit && events.length > limit) events.splice(0, events.length - limit);
    const identity = transcriptIdentityFromData(data, session);
    tailCache.set(sessionId, {
      threadId: identity.threadId,
      logPath: identity.logPath,
      liveCursor: typeof data.live_cursor === "string" && data.live_cursor ? data.live_cursor : null,
      hasOlder: Boolean(data.has_older),
      busy: Boolean(data.busy),
      queueLen: Number.isFinite(Number(data.queue_len)) ? Number(data.queue_len) : 0,
      token: data.token || null,
      events,
    });
  }

  function appendTailSnapshotEvents(tailCache, sessionIndex, sessionId, events, { session = null, identityData = null, liveCursor: nextLiveCursor, busy, queueLen, token, maxEvents = 0 } = {}) {
    if (!sessionId || !events || !events.length) return;
    const current = tailCache.get(sessionId);
    const list = current && Array.isArray(current.events) ? current.events.slice() : [];
    for (const ev of events) {
      const norm = normalizeTailEvent(ev);
      if (!norm) continue;
      list.push(norm);
    }
    const limit = Math.max(0, Number(maxEvents) || 0);
    if (limit && list.length > limit) list.splice(0, list.length - limit);
    const meta = session || sessionIndex.get(sessionId) || null;
    const identity = transcriptIdentityFromData(identityData, meta || current || null);
    tailCache.set(sessionId, {
      threadId: identity.threadId,
      logPath: identity.logPath,
      liveCursor: typeof nextLiveCursor === "string" && nextLiveCursor ? nextLiveCursor : current ? current.liveCursor : null,
      hasOlder: current ? Boolean(current.hasOlder) : false,
      busy: typeof busy === "boolean" ? busy : current ? Boolean(current.busy) : false,
      queueLen: Number.isFinite(Number(queueLen)) ? Number(queueLen) : current ? Number(current.queueLen || 0) : 0,
      token: token !== undefined ? token : current ? current.token : null,
      events: list,
    });
  }

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`transcript dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`transcript dependency missing: ${name}`);
    return value;
  }

  function createOlderLoadRuntime(options = {}) {
    const wrap = requireNode(options.olderWrap, "olderWrap");
    const button = requireNode(options.olderButton, "olderButton");
    const error = requireNode(options.olderError, "olderError");
    const errorText = requireNode(options.olderErrorText, "olderErrorText");
    const AbortControllerCtor = requireFunction(options.AbortControllerCtor, "AbortControllerCtor");
    const nowMs = requireFunction(options.nowMs, "nowMs");
    const autoCooldownMs = Math.max(0, Number(options.autoCooldownMs) || 0);
    let hasMore = false;
    let isLoading = false;
    let requestId = 0;
    let controller = null;
    let cancelOnScroll = true;
    let autoTriggerAt = 0;

    function snapshot() {
      return Object.freeze({ hasMore, isLoading, requestId, cancelOnScroll, hasController: Boolean(controller) });
    }

    function clearError() {
      error.style.display = "none";
      errorText.textContent = "";
    }

    function showError(message = "Couldn’t load older messages.") {
      errorText.textContent = String(message || "Couldn’t load older messages.");
      error.style.display = "flex";
    }

    function setState({ hasMore: nextHasMore, isLoading: nextLoading } = {}) {
      hasMore = Boolean(nextHasMore);
      isLoading = Boolean(nextLoading);
      wrap.style.display = hasMore ? "flex" : "none";
      button.disabled = isLoading;
      button.textContent = isLoading ? "Loading..." : "Load older messages";
      if (isLoading || !hasMore) clearError();
      return snapshot();
    }

    function resetAutoTrigger() {
      autoTriggerAt = 0;
    }

    function markAutoTrigger() {
      const now = Number(nowMs()) || 0;
      if (now - autoTriggerAt < autoCooldownMs) return false;
      autoTriggerAt = now;
      return true;
    }

    function beginLoad({ cancelOnScroll: nextCancelOnScroll = true } = {}) {
      requestId += 1;
      const ctl = new AbortControllerCtor();
      controller = ctl;
      cancelOnScroll = Boolean(nextCancelOnScroll);
      setState({ hasMore, isLoading: true });
      return Object.freeze({ requestId, controller: ctl, signal: ctl.signal });
    }

    function isCurrent(load) {
      const id = load && typeof load === "object" ? load.requestId : load;
      return id === requestId;
    }

    function finishLoad(load) {
      if (load && typeof load === "object" && controller === load.controller) controller = null;
      cancelOnScroll = true;
      return snapshot();
    }

    function invalidate() {
      if (!isLoading && !controller) return snapshot();
      requestId += 1;
      if (controller) {
        const ctl = controller;
        controller = null;
        try {
          ctl.abort();
        } catch (_) {}
      }
      cancelOnScroll = true;
      if (isLoading) setState({ hasMore, isLoading: false });
      return snapshot();
    }

    function shouldCancelOnScroll() {
      return Boolean(isLoading && cancelOnScroll);
    }

    return Object.freeze({
      beginLoad,
      clearError,
      finishLoad,
      invalidate,
      isCurrent,
      markAutoTrigger,
      resetAutoTrigger,
      setState,
      shouldCancelOnScroll,
      showError,
      snapshot,
    });
  }

  function createChatSearchAllRuntime(options = {}) {
    const setTimeoutFn = requireFunction(options.setTimeout, "setTimeout");
    const clearTimeoutFn = requireFunction(options.clearTimeout, "clearTimeout");
    const AbortControllerCtor = requireFunction(options.AbortControllerCtor, "AbortControllerCtor");
    const debounceMs = Math.max(0, Number(options.debounceMs) || 0);
    let count = null;
    let truncated = false;
    let hint = "";
    let requestId = 0;
    let abortController = null;
    let timer = null;

    function snapshot() {
      return Object.freeze({
        count,
        truncated,
        hint,
        requestId,
        hasAbort: Boolean(abortController),
        hasTimer: Boolean(timer),
      });
    }

    function abortActive() {
      if (!abortController) return;
      const ctl = abortController;
      abortController = null;
      try {
        ctl.abort();
      } catch (_) {}
    }

    function clearTimer() {
      if (!timer) return;
      clearTimeoutFn(timer);
      timer = null;
    }

    function reset() {
      count = null;
      truncated = false;
      hint = "";
      requestId += 1;
      clearTimer();
      abortActive();
      return snapshot();
    }

    function schedule(query, callback) {
      const run = requireFunction(callback, "callback");
      const cleanQuery = String(query || "").trim();
      reset();
      if (!cleanQuery) return Object.freeze({ scheduled: false, requestId, query: "" });
      const reqId = requestId;
      timer = setTimeoutFn(() => {
        timer = null;
        if (reqId !== requestId) return;
        run(cleanQuery);
      }, debounceMs);
      return Object.freeze({ scheduled: true, requestId: reqId, query: cleanQuery });
    }

    function beginRequest() {
      requestId += 1;
      abortActive();
      const ctl = new AbortControllerCtor();
      abortController = ctl;
      return Object.freeze({ requestId, controller: ctl, signal: ctl.signal });
    }

    function isCurrent(request) {
      return Boolean(request && request.requestId === requestId);
    }

    function completeRequest(request, result = {}) {
      if (!isCurrent(request)) return false;
      count = Number.isFinite(Number(result.count)) ? Number(result.count) : 0;
      truncated = Boolean(result.truncated);
      hint = String(result.hint || "");
      return true;
    }

    function failRequest(request) {
      if (!isCurrent(request)) return false;
      count = null;
      truncated = false;
      hint = "";
      return true;
    }

    function finishRequest(request) {
      if (request && abortController === request.controller) abortController = null;
      return snapshot();
    }

    function dispose() {
      return reset();
    }

    return Object.freeze({
      beginRequest,
      completeRequest,
      dispose,
      failRequest,
      finishRequest,
      isCurrent,
      reset,
      schedule,
      snapshot,
    });
  }

  window.CodoxearTranscript = Object.freeze({
    normalizeTailEvent,
    normalizeTranscriptState,
    transcriptKey,
    transcriptSnapshotFromData,
    transcriptIdentityFromData,
    tailCacheMatchesSession,
    rememberTailSnapshot,
    appendTailSnapshotEvents,
    createOlderLoadRuntime,
    createChatSearchAllRuntime,
  });
})();
