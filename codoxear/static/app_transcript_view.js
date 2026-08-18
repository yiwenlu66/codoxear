
  const VIEW_STATES = Object.freeze({
    LIVE: "LIVE",
    BROWSING: "BROWSING",
    LOADING_OLDER: "LOADING_OLDER",
    REPLACING: "REPLACING",
  });

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`transcript view dependency missing: ${name}`);
    return value;
  }

  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`transcript view dependency missing: ${name}`);
    return value;
  }

  // This is the policy boundary for the transcript. Rendering, scrolling, and
  // older-page mechanics remain in their focused runtimes; this controller is
  // the only code allowed to decide when each mechanic may mutate the view.
  function createTranscriptViewController(options = {}) {
    const root = requireObject(options.root, "root");
    const el = requireFunction(options.el, "el");
    const bottomSentinel = requireObject(options.bottomSentinel, "bottomSentinel");
    const messageRows = requireObject(options.messageRows, "messageRows");
    const transcript = requireObject(options.transcript, "transcript");
    const getMessageRowDeps = requireFunction(options.getMessageRowDeps, "getMessageRowDeps");
    const getSelectedSessionId = requireFunction(options.getSelectedSessionId, "getSelectedSessionId");
    const policyRuntime = requireObject(options.policyRuntime, "policyRuntime");
    const afterReplace = typeof options.afterReplace === "function" ? options.afterReplace : null;
    const domRuntime = requireObject(policyRuntime.domRuntime, "policyRuntime.domRuntime");
    const scrollRuntime = requireObject(policyRuntime.scrollRuntime, "policyRuntime.scrollRuntime");
    const setOlderState = requireFunction(policyRuntime.setOlderState, "policyRuntime.setOlderState");
    const getScrollTop = requireFunction(policyRuntime.getScrollTop, "policyRuntime.getScrollTop");
    const safeMakeRow = (event, rowOptions) => {
      const rowDeps = getMessageRowDeps();
      return requireFunction(messageRows.safeMakeRow, "messageRows.safeMakeRow")(event, rowOptions, {
        el: rowDeps?.el,
        chatMarkdownHtmlCached: rowDeps?.chatMarkdownHtmlCached,
        upgradeCandidateFileRefs: rowDeps?.upgradeCandidateFileRefs,
        time24: rowDeps?.time24,
        iconSvg: rowDeps?.iconSvg,
        copyToClipboard: rowDeps?.copyToClipboard,
        setToast: rowDeps?.setToast,
        chatAssistantDedupeKey: rowDeps?.chatAssistantDedupeKey,
        setTimeout: rowDeps?.setTimeout,
        consoleError: rowDeps?.consoleError,
        selectedSessionId: getSelectedSessionId(),
      });
    };

    function renderedMessageRows() {
      return requireFunction(messageRows.renderedMessageRows, "messageRows.renderedMessageRows")(root);
    }

    const renderRuntime = requireFunction(transcript.createTranscriptRenderRuntime, "transcript.createTranscriptRenderRuntime")({
      document: options.document,
      bottomSentinel,
      root,
      safeMakeRow,
      normalizeEvents: options.renderRuntime?.normalizeEvents,
      consumePendingUserIfMatches: options.renderRuntime?.consumePendingUserIfMatches,
      isDuplicateEvent: options.renderRuntime?.isDuplicateEvent,
      isAdjacentAssistantDuplicateEvent: options.renderRuntime?.isAdjacentAssistantDuplicateEvent,
      markEventSeen: options.renderRuntime?.markEventSeen,
      markFirstPaint: options.renderRuntime?.markFirstPaint,
      restorePendingRows: options.renderRuntime?.restorePendingRows,
      resetRecentEvents: options.renderRuntime?.resetRecentEvents,
      setOlderState: options.renderRuntime?.setOlderState,
      firstVisibleMessageRow: options.renderRuntime?.firstVisibleMessageRow,
      getScrollTop: options.renderRuntime?.getScrollTop,
      getSelectedSessionId: options.renderRuntime?.getSelectedSessionId,
      domRuntime: options.renderRuntime?.domRuntime,
      scrollRuntime: options.renderRuntime?.scrollRuntime,
      typingRowRuntime: options.renderRuntime?.typingRowRuntime,
      historySlackRows: options.renderRuntime?.historySlackRows,
    });

    let currentState = VIEW_STATES.LIVE;
    let historyCursor = null;
    let hasMore = false;
    let queuedLiveEvents = [];

    function setHistory({ cursor = historyCursor, nextHasMore = hasMore } = {}) {
      historyCursor = typeof cursor === "string" && cursor ? cursor : null;
      hasMore = Boolean(nextHasMore && historyCursor);
      setOlderState({ hasMore, isLoading: currentState === VIEW_STATES.LOADING_OLDER });
    }

    function enter(state) {
      currentState = state;
      setOlderState({ hasMore, isLoading: currentState === VIEW_STATES.LOADING_OLDER });
      return state;
    }

    function appendNow(events) {
      let changed = false;
      for (const event of events) changed = renderRuntime.appendEvent(event) || changed;
      return changed;
    }

    function appendEvents(events) {
      const list = Array.isArray(events) ? events : [];
      if (!list.length) return false;
      // A live delta arriving while an older page is in flight must not be
      // dropped merely because prepend owns the viewport for that moment.
      if (currentState === VIEW_STATES.LOADING_OLDER) {
        queuedLiveEvents.push(...list);
        return false;
      }
      if (currentState === VIEW_STATES.REPLACING) return false;
      return appendNow(list);
    }

    function flushQueuedLiveEvents() {
      if (!queuedLiveEvents.length) return false;
      const queued = queuedLiveEvents;
      queuedLiveEvents = [];
      return appendNow(queued);
    }

    function beginOlderLoad() {
      if (currentState !== VIEW_STATES.BROWSING || !hasMore) return false;
      enter(VIEW_STATES.LOADING_OLDER);
      return true;
    }

    function prependEvents(events, { cursor = historyCursor, nextHasMore = hasMore, preserveViewport = true } = {}) {
      if (currentState !== VIEW_STATES.LOADING_OLDER) return false;
      const changed = renderRuntime.prependOlderEvents(events, { preserveViewport });
      enter(VIEW_STATES.BROWSING);
      setHistory({ cursor, nextHasMore });
      flushQueuedLiveEvents();
      return changed;
    }

    function olderLoadFailed() {
      if (currentState !== VIEW_STATES.LOADING_OLDER) return false;
      enter(VIEW_STATES.BROWSING);
      flushQueuedLiveEvents();
      return true;
    }

    function replaceWith(events, { cursor = null, nextHasMore = false, preserveScroll = false, detached = false } = {}) {
      enter(VIEW_STATES.REPLACING);
      queuedLiveEvents = [];
      setHistory({ cursor, nextHasMore });
      const changed = detached
        ? renderRuntime.renderDetachedTranscriptWindow(events, { hasMore: Boolean(nextHasMore && cursor) })
        : renderRuntime.renderTranscript(events, { preserveScroll });
      if (!detached) {
        scrollRuntime.enableAutoScroll();
        scrollRuntime.markLiveTail();
        scrollRuntime.scheduleScrollToBottom({ double: true });
      }
      enter(detached ? VIEW_STATES.BROWSING : VIEW_STATES.LIVE);
      // DOM replacement detaches store-projected rows (typing/subagent activity);
      // their owner re-projects from the state authority after every rebuild.
      if (typeof afterReplace === "function") afterReplace();
      return changed;
    }

    function replaceWithPlaceholder(renderPlaceholder, { cursor = null, nextHasMore = false } = {}) {
      enter(VIEW_STATES.REPLACING);
      queuedLiveEvents = [];
      setHistory({ cursor, nextHasMore });
      // renderTranscript owns event/decorations reset; an empty replacement is
      // still a replacement and is the only path that may clear the DOM.
      renderRuntime.renderTranscript([], { preserveScroll: false });
      requireFunction(renderPlaceholder, "renderPlaceholder")();
      scrollRuntime.enableAutoScroll();
      scrollRuntime.markLiveTail();
      enter(VIEW_STATES.LIVE);
      if (typeof afterReplace === "function") afterReplace();
    }

    function replaceWithLoading({ cursor = null, nextHasMore = false } = {}) {
      replaceWithPlaceholder(() => {
        const row = el("div", { class: "msg-row assistant typing-row transcript-loading-row" });
        row.dataset.role = "assistant";
        row.appendChild(el("div", { class: "msg assistant loading", role: "status", "aria-live": "polite", text: "Loading transcript…" }));
        root.insertBefore(row, bottomSentinel);
      }, { cursor, nextHasMore });
    }

    function showLoadError({ message, onRetry }) {
      for (const row of Array.from(root.querySelectorAll(".transcript-error-row"))) row.remove();
      const row = el("div", { class: "msg-row assistant typing-row transcript-error-row" });
      row.dataset.role = "assistant";
      const bubble = el("div", { class: "msg assistant error transcript-error", role: "alert" });
      bubble.appendChild(el("span", { class: "transcriptErrorText", text: message }));
      const retryBtn = el("button", {
        class: "icon-btn text-btn transcriptRetryBtn",
        type: "button",
        text: "Retry",
        title: "Retry loading this transcript",
        "aria-label": "Retry loading this transcript",
      });
      retryBtn.onclick = onRetry;
      bubble.appendChild(retryBtn);
      row.appendChild(bubble);
      root.insertBefore(row, bottomSentinel);
    }

    function scrollToBottom({ force = false, events = null, cursor = null, nextHasMore = false } = {}) {
      if (force && Array.isArray(events)) return replaceWith(events, { cursor, nextHasMore });
      if (currentState === VIEW_STATES.REPLACING) return false;
      scrollRuntime.enableAutoScroll();
      scrollRuntime.markLiveTail();
      scrollRuntime.scheduleScrollToBottom({ syncJump: true });
      enter(VIEW_STATES.LIVE);
      flushQueuedLiveEvents();
      return true;
    }

    function observeScroll(method, event) {
      const action = requireFunction(scrollRuntime[method], `scrollRuntime.${method}`);
      // The scroll runtime may synchronously request an older page at the top
      // edge. Move policy into BROWSING first so that request can legally
      // transition to LOADING_OLDER instead of being rejected as LIVE.
      if (currentState === VIEW_STATES.LIVE && !scrollRuntime.shouldStickToBottom()) enter(VIEW_STATES.BROWSING);
      const result = action(event);
      if (currentState === VIEW_STATES.REPLACING || currentState === VIEW_STATES.LOADING_OLDER) return result;
      enter(scrollRuntime.shouldStickToBottom() ? VIEW_STATES.LIVE : VIEW_STATES.BROWSING);
      return result;
    }

    function state() {
      const scroll = scrollRuntime.snapshot();
      return Object.freeze({
        state: currentState,
        scrollTop: Number(getScrollTop()) || 0,
        renderedAtLiveTail: Boolean(scroll.renderedAtLiveTail),
        hasMore,
      });
    }

    return Object.freeze({
      VIEW_STATES,
      safeMakeRow,
      renderedMessageRows,
      loadedUserMessageRows: () => messageRows.loadedUserMessageRows(root),
      loadedCopyMessageRows: () => messageRows.loadedCopyMessageRows(root),
      rowSearchText: (row) => messageRows.rowSearchText(row),
      clearChatSearchMarks: () => messageRows.clearChatSearchMarks(renderedMessageRows()),
      applyChatSearchMarks: (matches, currentRow, query) => messageRows.applyChatSearchMarks(matches, currentRow, query),
      oldestRenderedHistoryCursor: () => messageRows.oldestRenderedHistoryCursor(renderedMessageRows()),
      firstVisibleMessageRow: (scrollTop) => messageRows.firstVisibleMessageRow(renderedMessageRows(), scrollTop),
      appendEvents,
      beginOlderLoad,
      olderLoadFailed,
      prependEvents,
      replaceWith,
      replaceWithPlaceholder,
      replaceWithLoading,
      scrollToBottom,
      observeScroll,
      setHistory,
      historyCursor: () => historyCursor,
      showLoadError,
      state,
    });
  }

export { createTranscriptViewController, VIEW_STATES };
