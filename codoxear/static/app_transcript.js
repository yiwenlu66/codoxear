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

  window.CodoxearTranscript = Object.freeze({
    normalizeTailEvent,
    normalizeTranscriptState,
    transcriptKey,
    transcriptSnapshotFromData,
    transcriptIdentityFromData,
    tailCacheMatchesSession,
    rememberTailSnapshot,
    appendTailSnapshotEvents,
  });
})();
