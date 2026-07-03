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

  function createTranscriptSlotRuntime(options = {}) {
    const sessionIndex = options.sessionIndex && typeof options.sessionIndex.get === "function" ? options.sessionIndex : new Map();
    const getSession = typeof options.getSession === "function" ? options.getSession : (sessionId) => sessionIndex.get(sessionId) || null;
    const sessionLookup = Object.freeze({ get: (sessionId) => getSession(sessionId) || null });
    const maxTailEvents = Math.max(0, Number(options.maxTailEvents) || 0);
    const slots = new Map();
    const tailCache = new Map();
    let active = defaultSlot();
    let liveCursor = null;

    function defaultSlot() {
      return { state: "pending_bind", threadId: null, logPath: null, key: null, epoch: 0, ignoredKey: null };
    }

    function cloneSlot(slot) {
      return slot ? { ...slot } : defaultSlot();
    }

    function getSlot(sessionId) {
      if (!sessionId) return defaultSlot();
      return cloneSlot(slots.get(sessionId) || defaultSlot());
    }

    function syncActiveSlot(sessionId) {
      active = getSlot(sessionId);
      return activeSnapshot();
    }

    function activeSnapshot() {
      return Object.freeze({ ...active, liveCursor });
    }

    function setActivePending() {
      active = defaultSlot();
      liveCursor = null;
      return activeSnapshot();
    }

    function setActiveFailed() {
      active = { ...defaultSlot(), state: "failed" };
      liveCursor = null;
      return activeSnapshot();
    }

    function setLiveCursor(value) {
      liveCursor = typeof value === "string" && value ? value : null;
      return activeSnapshot();
    }

    function clearLiveCursor() {
      return setLiveCursor(null);
    }

    function updateSlot(sessionId, data) {
      const prev = getSlot(sessionId);
      const snap = transcriptSnapshotFromData(data);
      if (prev.state === "pending_bind" && prev.ignoredKey && snap.state === "bound" && snap.key === prev.ignoredKey) {
        const next = { ...prev };
        if (sessionId) slots.set(sessionId, next);
        return Object.freeze({ previous: prev, current: cloneSlot(next), resetPending: false, ignoredStaleBound: true });
      }
      let epoch = prev.epoch;
      let resetPending = false;
      let ignoredKey = null;
      if (snap.state === "pending_bind") {
        if (prev.state === "bound") {
          epoch += 1;
          resetPending = true;
          ignoredKey = prev.key;
        } else {
          ignoredKey = prev.ignoredKey || null;
        }
      } else if (prev.state === "bound" && prev.key !== snap.key) {
        epoch += 1;
        resetPending = true;
      }
      const next = { ...snap, epoch, ignoredKey };
      if (sessionId) slots.set(sessionId, next);
      return Object.freeze({ previous: prev, current: cloneSlot(next), resetPending, ignoredStaleBound: false });
    }

    function beginRenewal(sessionId) {
      if (!sessionId) return null;
      const prev = getSlot(sessionId);
      const next = {
        state: "pending_bind",
        threadId: null,
        logPath: null,
        key: null,
        epoch: Number(prev.epoch || 0) + 1,
        ignoredKey: prev.state === "bound" ? prev.key : prev.ignoredKey || null,
      };
      slots.set(sessionId, next);
      return Object.freeze({ previous: prev, current: cloneSlot(next), resetPending: true });
    }

    function deleteSession(sessionId) {
      slots.delete(sessionId);
      tailCache.delete(sessionId);
    }

    function getTailCache(sessionId) {
      return tailCache.get(sessionId) || null;
    }

    function deleteTailCache(sessionId) {
      return tailCache.delete(sessionId);
    }

    function rememberTail(sessionId, session, data) {
      return rememberTailSnapshot(tailCache, sessionId, session, data, maxTailEvents);
    }

    function appendTailEvents(sessionId, events, options = {}) {
      return appendTailSnapshotEvents(tailCache, sessionLookup, sessionId, events, { ...options, maxEvents: maxTailEvents });
    }

    function snapshot() {
      return Object.freeze({ active: activeSnapshot(), slotCount: slots.size, tailCacheCount: tailCache.size });
    }

    return Object.freeze({
      activeSnapshot,
      appendTailEvents,
      beginRenewal,
      clearLiveCursor,
      deleteSession,
      deleteTailCache,
      getSlot,
      getTailCache,
      rememberTail,
      setActiveFailed,
      setActivePending,
      setLiveCursor,
      snapshot,
      syncActiveSlot,
      tailCacheMatchesSession,
      updateSlot,
    });
  }

  function createTypingRowRuntime(options = {}) {
    const root = requireNode(options.root, "root");
    const bottomSentinel = requireNode(options.bottomSentinel, "bottomSentinel");
    const el = requireFunction(options.el, "el");
    const shouldAutoScroll = requireFunction(options.shouldAutoScroll, "shouldAutoScroll");
    const scheduleScrollToBottom = requireFunction(options.scheduleScrollToBottom, "scheduleScrollToBottom");
    let typingRow = null;

    function ensureRow() {
      if (typingRow && typingRow.isConnected) return typingRow;
      const row = el("div", { class: "msg-row assistant typing-row" });
      row.dataset.role = "assistant";
      const bubble = el("div", { class: "msg assistant typing" });
      const dots = el("div", { class: "typingDots", "aria-label": "Running", title: "Running" }, [
        el("span", { class: "typingDot" }),
        el("span", { class: "typingDot" }),
        el("span", { class: "typingDot" }),
      ]);
      bubble.appendChild(dots);
      row.appendChild(bubble);
      typingRow = row;
      return row;
    }

    function anchor() {
      return typingRow && typingRow.isConnected ? typingRow : bottomSentinel;
    }

    function setVisible(show) {
      if (!show) {
        if (typingRow && typingRow.isConnected && typeof typingRow.remove === "function") typingRow.remove();
        return snapshot();
      }
      const row = ensureRow();
      if (!row.isConnected) {
        root.insertBefore(row, bottomSentinel);
      } else if (row.nextSibling !== bottomSentinel) {
        root.insertBefore(row, bottomSentinel);
      }
      if (shouldAutoScroll()) scheduleScrollToBottom();
      return snapshot();
    }

    function reset() {
      if (typingRow && typingRow.isConnected && typeof typingRow.remove === "function") typingRow.remove();
      typingRow = null;
      return snapshot();
    }

    function snapshot() {
      return Object.freeze({ connected: Boolean(typingRow && typingRow.isConnected) });
    }

    return Object.freeze({
      anchor,
      reset,
      setVisible,
      snapshot,
    });
  }

  function normalizedTranscriptEvents(events, options = {}) {
    const eventKey = requireFunction(options.eventKey, "eventKey");
    const consumePending = Boolean(options.consumePending);
    const takePendingMatch = consumePending ? requireFunction(options.takePendingMatch, "takePendingMatch") : null;
    const selectedSessionId = options.selectedSessionId;
    const msgs = [];
    const seen = new Set();
    for (const ev of events || []) {
      if (!ev || (ev.role !== "user" && ev.role !== "assistant")) continue;
      if (consumePending) takePendingMatch(ev, selectedSessionId, { allowUntimedCommit: false });
      const key = eventKey(ev);
      if (key && seen.has(key)) continue;
      if (key) seen.add(key);
      msgs.push(ev);
    }
    return msgs;
  }

  function createTranscriptRenderRuntime(options = {}) {
    const root = requireNode(options.root, "root");
    const bottomSentinel = requireNode(options.bottomSentinel, "bottomSentinel");
    const documentLike = requireNode(options.document, "document");
    const safeMakeRow = requireFunction(options.safeMakeRow, "safeMakeRow");
    const normalizeEvents = requireFunction(options.normalizeEvents, "normalizeEvents");
    const consumePendingUserIfMatches = requireFunction(options.consumePendingUserIfMatches, "consumePendingUserIfMatches");
    const isDuplicateEvent = requireFunction(options.isDuplicateEvent, "isDuplicateEvent");
    const isAdjacentAssistantDuplicateEvent = requireFunction(options.isAdjacentAssistantDuplicateEvent, "isAdjacentAssistantDuplicateEvent");
    const markEventSeen = requireFunction(options.markEventSeen, "markEventSeen");
    const markFirstPaint = requireFunction(options.markFirstPaint, "markFirstPaint");
    const renderRecoveryPanel = requireFunction(options.renderRecoveryPanel, "renderRecoveryPanel");
    const restorePendingRows = requireFunction(options.restorePendingRows, "restorePendingRows");
    const resetRecentEvents = requireFunction(options.resetRecentEvents, "resetRecentEvents");
    const setOlderState = requireFunction(options.setOlderState, "setOlderState");
    const firstVisibleMessageRow = requireFunction(options.firstVisibleMessageRow, "firstVisibleMessageRow");
    const getScrollTop = requireFunction(options.getScrollTop, "getScrollTop");
    const getSelectedSessionId = requireFunction(options.getSelectedSessionId, "getSelectedSessionId");
    const domRuntime = requireNode(options.domRuntime, "domRuntime");
    requireFunction(domRuntime.clear, "domRuntime.clear");
    requireFunction(domRuntime.rebuildDecorations, "domRuntime.rebuildDecorations");
    requireFunction(domRuntime.trimRenderedRows, "domRuntime.trimRenderedRows");
    const scrollRuntime = requireNode(options.scrollRuntime, "scrollRuntime");
    requireFunction(scrollRuntime.shouldStickToBottom, "scrollRuntime.shouldStickToBottom");
    requireFunction(scrollRuntime.snapshot, "scrollRuntime.snapshot");
    requireFunction(scrollRuntime.syncJumpButton, "scrollRuntime.syncJumpButton");
    requireFunction(scrollRuntime.scheduleScrollToBottom, "scrollRuntime.scheduleScrollToBottom");
    requireFunction(scrollRuntime.markLiveTail, "scrollRuntime.markLiveTail");
    requireFunction(scrollRuntime.disableAutoScroll, "scrollRuntime.disableAutoScroll");
    requireFunction(scrollRuntime.setRenderedAtLiveTail, "scrollRuntime.setRenderedAtLiveTail");
    requireFunction(scrollRuntime.setScrollTop, "scrollRuntime.setScrollTop");
    const typingRowRuntime = requireNode(options.typingRowRuntime, "typingRowRuntime");
    requireFunction(typingRowRuntime.anchor, "typingRowRuntime.anchor");
    const historySlackRows = Math.max(1, Math.floor(Number(options.historySlackRows) || 1));

    function fragmentFor(events, { markSeen = false, pending = false } = {}) {
      const frag = documentLike.createDocumentFragment();
      for (const ev of events) {
        const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : pending && ev.pending ? Date.now() / 1000 : null;
        if (markSeen) markEventSeen(ev);
        frag.appendChild(safeMakeRow(ev, { ts, pending }).row);
      }
      return frag;
    }

    function appendEvent(ev) {
      if (!ev || (ev.role !== "user" && ev.role !== "assistant")) return false;
      if (consumePendingUserIfMatches(ev)) return false;
      if (isDuplicateEvent(ev)) return false;
      if (isAdjacentAssistantDuplicateEvent(ev)) {
        markEventSeen(ev);
        return false;
      }
      const pending = Boolean(ev.pending);
      const stick = pending || scrollRuntime.shouldStickToBottom();
      if (!pending && !scrollRuntime.snapshot().renderedAtLiveTail) {
        markEventSeen(ev);
        scrollRuntime.syncJumpButton();
        return false;
      }
      const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : ev.pending ? Date.now() / 1000 : null;
      const { row } = safeMakeRow(ev, { ts, pending });
      root.insertBefore(row, typingRowRuntime.anchor());
      domRuntime.trimRenderedRows({ fromTop: stick });
      domRuntime.rebuildDecorations({ preserveScroll: false });
      renderRecoveryPanel(getSelectedSessionId());
      if (!ev.pending) markFirstPaint();
      markEventSeen(ev);
      if (stick) scrollRuntime.scheduleScrollToBottom();
      scrollRuntime.syncJumpButton();
      return true;
    }

    function renderTranscript(events, { preserveScroll = false } = {}) {
      const selectedSessionId = getSelectedSessionId();
      const msgs = normalizeEvents(events, { consumePending: true });
      scrollRuntime.markLiveTail();
      domRuntime.clear();
      if (!msgs.length) {
        restorePendingRows(selectedSessionId);
        return false;
      }
      resetRecentEvents();
      root.insertBefore(fragmentFor(msgs, { markSeen: true, pending: false }), bottomSentinel);
      domRuntime.rebuildDecorations({ preserveScroll });
      restorePendingRows(selectedSessionId);
      return true;
    }

    function renderDetachedTranscriptWindow(events, { hasMore = false } = {}) {
      const msgs = normalizeEvents(events, { consumePending: false });
      scrollRuntime.disableAutoScroll();
      scrollRuntime.setRenderedAtLiveTail(false);
      domRuntime.clear();
      setOlderState({ hasMore: Boolean(hasMore), isLoading: false });
      if (!msgs.length) {
        scrollRuntime.syncJumpButton();
        return false;
      }
      resetRecentEvents();
      root.insertBefore(fragmentFor(msgs, { markSeen: true, pending: false }), bottomSentinel);
      domRuntime.rebuildDecorations({ preserveScroll: false });
      scrollRuntime.setScrollTop(1);
      scrollRuntime.syncJumpButton();
      return true;
    }

    function prependOlderEvents(allEvents, { preserveViewport = false } = {}) {
      const msgs = [];
      for (const ev of allEvents || []) {
        if (!ev || (ev.role !== "user" && ev.role !== "assistant")) continue;
        msgs.push(ev);
      }
      if (!msgs.length) return false;
      scrollRuntime.disableAutoScroll();
      const frag = fragmentFor(msgs, { markSeen: false, pending: false });
      const anchorRow = preserveViewport ? firstVisibleMessageRow() : null;
      const anchorOffset = anchorRow ? anchorRow.offsetTop - getScrollTop() : 0;
      const firstMsg = root.querySelector(".msg-row:not(.typing-row)");
      root.insertBefore(frag, firstMsg || typingRowRuntime.anchor());
      const wasAtLiveTail = scrollRuntime.snapshot().renderedAtLiveTail;
      if (!preserveViewport) scrollRuntime.setScrollTop(1);
      domRuntime.trimRenderedRows({ fromTop: false, maxRows: historySlackRows });
      if (wasAtLiveTail && scrollRuntime.snapshot().renderedAtLiveTail === false) {
        scrollRuntime.disableAutoScroll();
      }
      domRuntime.rebuildDecorations({ preserveScroll: false });
      if (preserveViewport && anchorRow && anchorRow.isConnected) {
        scrollRuntime.setScrollTop(Math.max(0, anchorRow.offsetTop - anchorOffset));
      } else {
        scrollRuntime.setScrollTop(1);
      }
      scrollRuntime.syncJumpButton();
      return true;
    }

    return Object.freeze({
      appendEvent,
      prependOlderEvents,
      renderDetachedTranscriptWindow,
      renderTranscript,
    });
  }

  function createTranscriptDomRuntime(options = {}) {
    const root = requireNode(options.root, "root");
    requireFunction(root.appendChild, "root.appendChild");
    requireFunction(root.insertBefore, "root.insertBefore");
    requireFunction(root.querySelectorAll, "root.querySelectorAll");
    const olderWrap = requireNode(options.olderWrap, "olderWrap");
    const bottomSentinel = requireNode(options.bottomSentinel, "bottomSentinel");
    const el = requireFunction(options.el, "el");
    const ymd = requireFunction(options.ymd, "ymd");
    const dayLabel = requireFunction(options.dayLabel, "dayLabel");
    const getRenderedRows = requireFunction(options.getRenderedRows, "getRenderedRows");
    const trimRenderedRowTargets = requireFunction(options.trimRenderedRowTargets, "trimRenderedRowTargets");
    const trimRowsBeforeViewportTargets = requireFunction(options.trimRowsBeforeViewportTargets, "trimRowsBeforeViewportTargets");
    const afterDecorate = requireFunction(options.afterDecorate, "afterDecorate");
    const scrollRuntime = requireNode(options.scrollRuntime, "scrollRuntime");
    requireFunction(scrollRuntime.captureScrollPosition, "scrollRuntime.captureScrollPosition");
    requireFunction(scrollRuntime.preserveScrollFrom, "scrollRuntime.preserveScrollFrom");
    requireFunction(scrollRuntime.snapshot, "scrollRuntime.snapshot");
    requireFunction(scrollRuntime.scheduleScrollToBottom, "scrollRuntime.scheduleScrollToBottom");
    requireFunction(scrollRuntime.syncJumpButton, "scrollRuntime.syncJumpButton");
    requireFunction(scrollRuntime.setRenderedAtLiveTail, "scrollRuntime.setRenderedAtLiveTail");
    const defaultWindowRows = Math.max(1, Math.floor(Number(options.defaultWindowRows) || 1));

    function clear() {
      root.innerHTML = "";
      root.appendChild(olderWrap);
      root.appendChild(bottomSentinel);
    }

    function rebuildDecorations({ preserveScroll = false } = {}) {
      const scrollPosition = scrollRuntime.captureScrollPosition();
      for (const n of Array.from(root.querySelectorAll(".day-sep"))) n.remove();
      const rows = getRenderedRows();
      let prevRole = null;
      let prevDay = null;
      let lastDay = null;
      for (const row of rows) {
        const role = row.classList.contains("user") ? "user" : "assistant";
        const ts = Number(row.dataset.ts || "0");
        const day = ts ? ymd(new Date(ts * 1000)) : null;
        row.classList.remove("grouped");
        if (prevRole === role && prevDay && day && prevDay === day) row.classList.add("grouped");
        prevRole = role;
        prevDay = day;
        if (day && day !== lastDay) {
          const d = new Date(ts * 1000);
          const sep = el("div", { class: "day-sep", text: dayLabel(d) });
          sep.dataset.day = day;
          root.insertBefore(sep, row);
          lastDay = day;
        }
      }
      if (preserveScroll) scrollRuntime.preserveScrollFrom(scrollPosition);
      if (scrollRuntime.snapshot().autoScroll) scrollRuntime.scheduleScrollToBottom();
      scrollRuntime.syncJumpButton();
      afterDecorate();
    }

    function trimRenderedRows({ fromTop, maxRows = defaultWindowRows } = {}) {
      const targets = trimRenderedRowTargets(getRenderedRows(), fromTop, maxRows, defaultWindowRows);
      if (!targets.length) return 0;
      for (const row of targets) row.remove();
      scrollRuntime.setRenderedAtLiveTail(Boolean(fromTop));
      return targets.length;
    }

    function trimRowsBeforeViewport({ maxRows = defaultWindowRows, viewportTop = 0 } = {}) {
      const targets = trimRowsBeforeViewportTargets(getRenderedRows(), maxRows, defaultWindowRows, viewportTop);
      if (!targets.length) return 0;
      for (const row of targets) row.remove();
      return targets.length;
    }

    return Object.freeze({
      clear,
      rebuildDecorations,
      trimRenderedRows,
      trimRowsBeforeViewport,
    });
  }

  function createTranscriptScrollRuntime(options = {}) {
    const chat = requireNode(options.chat, "chat");
    const jumpButton = requireNode(options.jumpButton, "jumpButton");
    const timeChip = requireNode(options.timeChip, "timeChip");
    const requestAnimationFrameFn = requireFunction(options.requestAnimationFrame, "requestAnimationFrame");
    const hasSelection = requireFunction(options.hasSelection, "hasSelection");
    const isSearchOpen = requireFunction(options.isSearchOpen, "isSearchOpen");
    const firstVisibleMessageRow = requireFunction(options.firstVisibleMessageRow, "firstVisibleMessageRow");
    const dayLabel = requireFunction(options.dayLabel, "dayLabel");
    const time24 = requireFunction(options.time24, "time24");
    const shouldCancelOlderLoad = requireFunction(options.shouldCancelOlderLoad, "shouldCancelOlderLoad");
    const cancelOlderLoad = requireFunction(options.cancelOlderLoad, "cancelOlderLoad");
    const autoLoadOlder = requireFunction(options.autoLoadOlder, "autoLoadOlder");
    const bottomThresholdPx = Math.max(0, Number(options.bottomThresholdPx) || 80);
    const olderTopTriggerPx = Math.max(0, Number(options.olderTopTriggerPx) || 0);
    const olderCancelPx = Math.max(0, Number(options.olderCancelPx) || 0);
    let autoScroll = true;
    let renderedAtLiveTail = true;
    let lastScrollTop = Number(chat.scrollTop) || 0;
    let touchY = null;

    function snapshot() {
      return Object.freeze({ autoScroll, renderedAtLiveTail, lastScrollTop });
    }

    function isNearBottom() {
      const scrollHeight = Number(chat.scrollHeight) || 0;
      const scrollTop = Number(chat.scrollTop) || 0;
      const clientHeight = Number(chat.clientHeight) || 0;
      return scrollHeight - (scrollTop + clientHeight) <= bottomThresholdPx;
    }

    function shouldStickToBottom() {
      return Boolean(renderedAtLiveTail && (autoScroll || isNearBottom()));
    }

    function shouldAutoScrollOrNearBottom() {
      return Boolean(autoScroll || isNearBottom());
    }

    function syncVisibleTimeIndicator() {
      if (!hasSelection() || isSearchOpen() || shouldStickToBottom()) {
        timeChip.style.display = "none";
        timeChip.textContent = "";
        return Object.freeze({ visible: false, text: "" });
      }
      const row = firstVisibleMessageRow();
      const ts = row ? Number(row.dataset && row.dataset.ts ? row.dataset.ts : "0") : 0;
      if (!Number.isFinite(ts) || ts <= 0) {
        timeChip.style.display = "none";
        timeChip.textContent = "";
        return Object.freeze({ visible: false, text: "" });
      }
      const d = new Date(ts * 1000);
      const text = `${dayLabel(d)} · ${time24(d)}`;
      timeChip.textContent = text;
      timeChip.style.display = "inline-flex";
      return Object.freeze({ visible: true, text });
    }

    function syncJumpButton() {
      jumpButton.style.display = shouldStickToBottom() ? "none" : "inline-flex";
      syncVisibleTimeIndicator();
      return snapshot();
    }

    function setAutoScroll(nextAutoScroll) {
      autoScroll = Boolean(nextAutoScroll);
      return snapshot();
    }

    function enableAutoScroll() {
      return setAutoScroll(true);
    }

    function disableAutoScroll() {
      return setAutoScroll(false);
    }

    function setRenderedAtLiveTail(nextRenderedAtLiveTail) {
      renderedAtLiveTail = Boolean(nextRenderedAtLiveTail);
      return snapshot();
    }

    function markLiveTail() {
      return setRenderedAtLiveTail(true);
    }

    function markDetachedWindow() {
      autoScroll = false;
      renderedAtLiveTail = false;
      return snapshot();
    }

    function setScrollTop(nextTop) {
      const top = Math.max(0, Number(nextTop) || 0);
      chat.scrollTop = top;
      lastScrollTop = Number(chat.scrollTop) || 0;
      return snapshot();
    }

    function scrollToBottom() {
      chat.scrollTop = Number(chat.scrollHeight) || 0;
      lastScrollTop = Number(chat.scrollTop) || 0;
      return snapshot();
    }

    function scheduleScrollToBottom({ double = false, syncJump = false } = {}) {
      requestAnimationFrameFn(() => {
        scrollToBottom();
        if (double) requestAnimationFrameFn(() => scrollToBottom());
        if (syncJump) syncJumpButton();
      });
    }

    function captureScrollPosition() {
      return Object.freeze({ top: Number(chat.scrollTop) || 0, height: Number(chat.scrollHeight) || 0 });
    }

    function preserveScrollFrom(position) {
      const top = position && Number.isFinite(Number(position.top)) ? Number(position.top) : 0;
      const height = position && Number.isFinite(Number(position.height)) ? Number(position.height) : 0;
      chat.scrollTop = top + ((Number(chat.scrollHeight) || 0) - height);
      return snapshot();
    }

    function reset({ scrollTop = 0 } = {}) {
      autoScroll = true;
      renderedAtLiveTail = true;
      setScrollTop(scrollTop);
      syncVisibleTimeIndicator();
      jumpButton.style.display = "none";
      return snapshot();
    }

    function maybeAutoLoadOlder() {
      if ((Number(chat.scrollTop) || 0) > olderTopTriggerPx) return false;
      autoLoadOlder();
      return true;
    }

    function handleScroll() {
      const cur = Number(chat.scrollTop) || 0;
      const delta = cur - lastScrollTop;
      lastScrollTop = cur;
      if (delta < 0) autoScroll = false;
      else if (isNearBottom()) autoScroll = true;
      if (shouldCancelOlderLoad() && cur > olderCancelPx) cancelOlderLoad();
      if (cur <= olderTopTriggerPx && delta <= 0) maybeAutoLoadOlder();
      syncJumpButton();
      return Object.freeze({ delta, ...snapshot() });
    }

    function handleWheel(event) {
      if (event && Number(event.deltaY) < 0) {
        autoScroll = false;
        syncJumpButton();
        maybeAutoLoadOlder();
      }
      return snapshot();
    }

    function handleTouchStart(event) {
      const touch = event && event.touches && event.touches[0];
      touchY = touch ? Number(touch.clientY) : null;
      return snapshot();
    }

    function handleTouchMove(event) {
      const touch = event && event.touches && event.touches[0];
      if (!touch || touchY === null) return snapshot();
      const y = Number(touch.clientY);
      const deltaY = y - touchY;
      touchY = y;
      if (deltaY > 0) {
        autoScroll = false;
        syncJumpButton();
        maybeAutoLoadOlder();
      }
      return snapshot();
    }

    return Object.freeze({
      captureScrollPosition,
      disableAutoScroll,
      enableAutoScroll,
      handleScroll,
      handleTouchMove,
      handleTouchStart,
      handleWheel,
      isNearBottom,
      markDetachedWindow,
      markLiveTail,
      maybeAutoLoadOlder,
      preserveScrollFrom,
      reset,
      scheduleScrollToBottom,
      scrollToBottom,
      setAutoScroll,
      setRenderedAtLiveTail,
      setScrollTop,
      shouldAutoScrollOrNearBottom,
      shouldStickToBottom,
      snapshot,
      syncJumpButton,
      syncVisibleTimeIndicator,
    });
  }

  function createTranscriptEventRuntime(options = {}) {
    const eventKey = requireFunction(options.eventKey, "eventKey");
    const pendingMatchKey = requireFunction(options.pendingMatchKey, "pendingMatchKey");
    const normalizePendingText = requireFunction(options.normalizePendingText, "normalizePendingText");
    const assistantDedupeKey = requireFunction(options.assistantDedupeKey, "assistantDedupeKey");
    const maxRecentEventKeys = Math.max(1, Number(options.maxRecentEventKeys) || 320);
    const recentEventKeys = [];
    const recentEventKeySet = new Set();
    let localEchoSeq = 0;
    let pendingUsers = [];

    function clonePending(item) {
      return item ? { ...item } : item;
    }

    function snapshot() {
      return Object.freeze({
        localEchoSeq,
        pendingCount: pendingUsers.length,
        recentEventKeys: recentEventKeys.slice(),
      });
    }

    function resetRecentEvents() {
      recentEventKeys.length = 0;
      recentEventKeySet.clear();
      return snapshot();
    }

    function markEventSeen(ev) {
      const key = eventKey(ev);
      if (!key) return false;
      if (recentEventKeySet.has(key)) return false;
      recentEventKeySet.add(key);
      recentEventKeys.push(key);
      if (recentEventKeys.length > maxRecentEventKeys) {
        const drop = recentEventKeys.splice(0, recentEventKeys.length - maxRecentEventKeys);
        for (const k of drop) recentEventKeySet.delete(k);
      }
      return true;
    }

    function isDuplicateEvent(ev) {
      const key = eventKey(ev);
      if (!key) return false;
      return recentEventKeySet.has(key);
    }

    function isAdjacentAssistantDuplicateEvent(ev, { renderedAtLiveTail = false, rows = [] } = {}) {
      if (!renderedAtLiveTail || !ev || ev.pending || ev.role !== "assistant") return false;
      const key = assistantDedupeKey(ev);
      if (!key) return false;
      const list = Array.isArray(rows) ? rows : [];
      const last = list.length ? list[list.length - 1] : null;
      return Boolean(last && last.dataset && last.dataset.role === "assistant" && last.dataset.assistantDedupeKey === key);
    }

    function nextLocalEchoId() {
      localEchoSeq += 1;
      return localEchoSeq;
    }

    function addPendingUser({ id = null, sessionId = "", epoch = 0, text = "", t0 = 0 } = {}) {
      const pendingId = Number.isFinite(Number(id)) && Number(id) > 0 ? Number(id) : nextLocalEchoId();
      if (pendingId > localEchoSeq) localEchoSeq = pendingId;
      const raw = String(text || "");
      const item = {
        id: pendingId,
        sessionId,
        epoch: Number(epoch || 0),
        key: pendingMatchKey(raw),
        loose: normalizePendingText(raw),
        t0: Number(t0 || 0),
        text: raw,
      };
      pendingUsers.push(item);
      return clonePending(item);
    }

    function pendingUsersForSession(sessionId, epoch) {
      const slotEpoch = Number(epoch || 0);
      return pendingUsers
        .filter((item) => item && item.sessionId === sessionId && Number(item.epoch || 0) === slotEpoch)
        .sort((a, b) => Number(a.t0 || 0) - Number(b.t0 || 0))
        .map(clonePending);
    }

    function dropPendingUsers(sessionId, predicate = null) {
      const pred = typeof predicate === "function" ? predicate : () => true;
      const kept = [];
      const dropped = [];
      for (const item of pendingUsers) {
        const match = Boolean(item && item.sessionId === sessionId && pred(clonePending(item)));
        if (match) dropped.push(clonePending(item));
        else kept.push(item);
      }
      pendingUsers = kept;
      return dropped;
    }

    function hasPendingForSession(sessionId) {
      return pendingUsers.some((pending) => pending && pending.sessionId === sessionId);
    }

    function takePendingUserMatch(ev, sessionId, epoch, { allowUntimedCommit = true } = {}) {
      if (!ev || ev.role !== "user" || ev.pending) return false;
      const slotEpoch = Number(epoch || 0);
      const key = pendingMatchKey(ev.text);
      const loose = normalizePendingText(ev.text);
      const evTs = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : null;
      const sameSlot = [];
      const exactCandidates = [];
      for (let i = 0; i < pendingUsers.length; i += 1) {
        const x = pendingUsers[i];
        if (!x || x.sessionId !== sessionId || Number(x.epoch || 0) !== slotEpoch) continue;
        const candidate = { i, x };
        sameSlot.push(candidate);
        if (x.key === key || x.loose === loose) exactCandidates.push(candidate);
      }
      const candidates = exactCandidates.length
        ? exactCandidates
        : sameSlot.filter(({ x }) => (evTs !== null ? evTs >= Number(x.t0 || 0) - 5 : allowUntimedCommit));
      if (!candidates.length) return false;
      let best = candidates[0];
      if (exactCandidates.length && evTs !== null) {
        let bestD = Math.abs(evTs - (best.x.t0 || evTs));
        for (const c of candidates.slice(1)) {
          const d = Math.abs(evTs - (c.x.t0 || evTs));
          if (d < bestD) {
            best = c;
            bestD = d;
          }
        }
      }
      const idx = best.i;
      if (idx < 0) return null;
      const match = pendingUsers[idx];
      pendingUsers.splice(idx, 1);
      return clonePending(match) || null;
    }

    return Object.freeze({
      addPendingUser,
      dropPendingUsers,
      hasPendingForSession,
      isAdjacentAssistantDuplicateEvent,
      isDuplicateEvent,
      markEventSeen,
      nextLocalEchoId,
      pendingUsersForSession,
      resetRecentEvents,
      snapshot,
      takePendingUserMatch,
    });
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

  function createLoadedChatSearchRuntime() {
    let open = false;
    let query = "";
    let matches = [];
    let index = -1;
    let loadingOlder = false;

    function normalizeQuery(value) {
      return String(value || "").trim().toLowerCase();
    }

    function snapshot() {
      return Object.freeze({ open, query, matches: matches.slice(), index, loadingOlder });
    }

    function setOpen(nextOpen) {
      open = Boolean(nextOpen);
      return snapshot();
    }

    function setLoadingOlder(nextLoading) {
      loadingOlder = Boolean(nextLoading);
      return snapshot();
    }

    function setQuery(value) {
      query = normalizeQuery(value);
      return query;
    }

    function clearMatches() {
      matches = [];
      index = -1;
      return snapshot();
    }

    function setMatches(nextMatches, { preserveCurrent = true } = {}) {
      const previous = preserveCurrent && index >= 0 ? matches[index] : null;
      matches = Array.isArray(nextMatches) ? nextMatches.filter(Boolean) : [];
      if (!matches.length) {
        index = -1;
        return snapshot();
      }
      const nextIndex = previous ? matches.indexOf(previous) : -1;
      index = nextIndex >= 0 ? nextIndex : 0;
      return snapshot();
    }

    function focusIndex(nextIndex) {
      if (!matches.length) {
        index = -1;
        return Object.freeze({ index, row: null, matches: [] });
      }
      const total = matches.length;
      index = ((Number(nextIndex) % total) + total) % total;
      return Object.freeze({ index, row: matches[index], matches: matches.slice() });
    }

    function ensureTargetRow(target, forcedQuery, compareRowsInDomOrder) {
      if (!target) return -1;
      target.dataset.searchForcedQuery = normalizeQuery(forcedQuery);
      if (!matches.includes(target)) {
        matches.push(target);
        if (typeof compareRowsInDomOrder === "function") matches.sort(compareRowsInDomOrder);
      }
      index = matches.indexOf(target);
      return index;
    }

    function reset() {
      open = false;
      query = "";
      loadingOlder = false;
      return clearMatches();
    }

    return Object.freeze({
      clearMatches,
      ensureTargetRow,
      focusIndex,
      reset,
      setLoadingOlder,
      setMatches,
      setOpen,
      setQuery,
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
    createTranscriptSlotRuntime,
    createTypingRowRuntime,
    normalizedTranscriptEvents,
    createTranscriptRenderRuntime,
    createTranscriptDomRuntime,
    createTranscriptScrollRuntime,
    createTranscriptEventRuntime,
    createOlderLoadRuntime,
    createLoadedChatSearchRuntime,
    createChatSearchAllRuntime,
  });
})();
