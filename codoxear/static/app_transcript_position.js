/* The reader's viewport position inside a transcript, per session.

   Two different things are remembered, because they are two different kinds of
   position:

   - "following the live tail" is an invariant, not a coordinate. It survives
     the content growing below the viewport (an image whose header dimensions
     arrive after paint, a row inserted after the last scroll) because the
     scroll runtime re-asserts the bottom on every content-size change.
   - a reading position is an *anchor*: the message row at the top of the
     viewport plus how far into that row the reader was. A pixel offset is not a
     position — the same transcript renders at a different height on another
     viewport, and any late layout change moves content under a fixed offset.

   This controller is the single writer of that remembered position: the
   lifecycle asks it to save on switch-out and restore after the render that the
   session is finally showing, and it keeps the anchor placed while the DOM
   settles. It stands down as soon as anyone else owns the viewport (a reader
   gesture, navigation's scrollIntoView, the platform's own anchoring, or
   another writer in the scroll runtime), so it can never fight a deliberate
   movement. */

  function positionRequireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`transcript position dependency missing: ${name}`);
    return value;
  }

  function positionRequireNode(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`transcript position dependency missing: ${name}`);
    return value;
  }

  function positionSessionId(sessionId) {
    return typeof sessionId === "string" ? sessionId.trim() : "";
  }

  function positionRowField(row, name) {
    const dataset = row && row.dataset ? row.dataset : null;
    const value = dataset ? dataset[name] : "";
    return typeof value === "string" ? value.trim() : "";
  }

  function createTranscriptPositionController(options = {}) {
    const chat = positionRequireNode(options.chat, "chat");
    const content = positionRequireNode(options.content, "content");
    const scrollRuntime = positionRequireNode(options.scrollRuntime, "scrollRuntime");
    const getSelectedSessionId = positionRequireFunction(options.getSelectedSessionId, "getSelectedSessionId");
    const renderedRows = positionRequireFunction(options.renderedRows, "renderedRows");
    const resolveAnchorWindow = positionRequireFunction(options.resolveAnchorWindow, "resolveAnchorWindow");
    const ResizeObserverCtor = typeof options.resizeObserver === "function" ? options.resizeObserver : null;
    positionRequireFunction(scrollRuntime.setScrollTop, "scrollRuntime.setScrollTop");
    positionRequireFunction(scrollRuntime.scrollToBottom, "scrollRuntime.scrollToBottom");
    positionRequireFunction(scrollRuntime.syncJumpButton, "scrollRuntime.syncJumpButton");
    positionRequireFunction(scrollRuntime.snapshot, "scrollRuntime.snapshot");
    positionRequireFunction(scrollRuntime.isNearBottom, "scrollRuntime.isNearBottom");
    positionRequireFunction(scrollRuntime.markLiveTail, "scrollRuntime.markLiveTail");
    positionRequireFunction(scrollRuntime.enableAutoScroll, "scrollRuntime.enableAutoScroll");
    positionRequireFunction(scrollRuntime.disableAutoScroll, "scrollRuntime.disableAutoScroll");
    positionRequireFunction(scrollRuntime.cancelScheduledBottomWrites, "scrollRuntime.cancelScheduledBottomWrites");
    positionRequireFunction(scrollRuntime.writeSeq, "scrollRuntime.writeSeq");
    positionRequireFunction(scrollRuntime.unownedScrollSeq, "scrollRuntime.unownedScrollSeq");

    const positions = new Map();
    // Bumped by every restore/cancel so an in-flight anchor materialization for
    // a stale generation cannot place a row in a transcript that moved on.
    let sequence = 0;
    let settle = null;

    const rowsNow = () => {
      const rows = renderedRows();
      return Array.isArray(rows) ? rows : [];
    };

    const chatTop = () => Number(chat.getBoundingClientRect().top) || 0;

    function rowRect(row) {
      return row && typeof row.getBoundingClientRect === "function" ? row.getBoundingClientRect() : null;
    }

    function anchorForRow(row) {
      const rect = rowRect(row);
      if (!rect) return null;
      return Object.freeze({
        messageId: positionRowField(row, "messageId"),
        cursor: positionRowField(row, "historyCursor"),
        offset: (Number(rect.top) || 0) - chatTop(),
      });
    }

    function firstVisibleRow(rows) {
      const top = chatTop();
      for (const row of rows) {
        const rect = rowRect(row);
        if (rect && Number(rect.bottom) > top + 1) return row;
      }
      return null;
    }

    function findRow(position) {
      const rows = rowsNow();
      const messageId = position && position.messageId ? position.messageId : "";
      const cursor = position && position.cursor ? position.cursor : "";
      if (messageId) {
        for (const row of rows) if (positionRowField(row, "messageId") === messageId) return row;
      }
      if (cursor) {
        for (const row of rows) if (positionRowField(row, "historyCursor") === cursor) return row;
      }
      return null;
    }

    // The saved position for a session: `{kind: "tail"}`, or an anchor naming
    // the row at the top of the viewport and how far into it the reader was.
    function save(sessionId) {
      const id = positionSessionId(sessionId);
      if (!id) return false;
      const snapshot = scrollRuntime.snapshot();
      const atTail = Boolean(snapshot.renderedAtLiveTail) && scrollRuntime.isNearBottom();
      if (atTail) {
        positions.set(id, Object.freeze({ kind: "tail" }));
        return true;
      }
      const anchor = anchorForRow(firstVisibleRow(rowsNow()));
      if (!anchor || (!anchor.messageId && !anchor.cursor)) {
        // Nothing identifiable is rendered (a loading placeholder, an error
        // row). Falling back to the tail is the only honest answer.
        positions.set(id, Object.freeze({ kind: "tail" }));
        return true;
      }
      positions.set(id, Object.freeze({
        kind: "anchor",
        messageId: anchor.messageId,
        cursor: anchor.cursor,
        offset: anchor.offset,
      }));
      return true;
    }

    function clear(sessionId) {
      return positions.delete(positionSessionId(sessionId));
    }

    function positionFor(sessionId) {
      return positions.get(positionSessionId(sessionId)) || null;
    }

    function stopSettling() {
      settle = null;
    }

    function cancel() {
      sequence += 1;
      stopSettling();
    }

    function placeAtRow(row, offset) {
      const rect = rowRect(row);
      if (!rect) return false;
      const target = (Number(chat.scrollTop) || 0) + ((Number(rect.top) || 0) - chatTop()) - (Number(offset) || 0);
      scrollRuntime.setScrollTop(Math.max(0, target));
      return true;
    }

    function writeTail() {
      scrollRuntime.enableAutoScroll();
      scrollRuntime.markLiveTail();
      scrollRuntime.scrollToBottom();
      scrollRuntime.syncJumpButton();
    }

    // Re-establish the anchor while the rendered content changes size. Three
    // things end it, and all three mean somebody else now owns the viewport:
    // the reader moved it, another writer moved it, or the row it names is gone
    // (the transcript was replaced). None of them is a timing assumption.
    function reassert() {
      if (!settle) return;
      const state = settle;
      if (state.sequence !== sequence) return stopSettling();
      if (getSelectedSessionId() !== state.sessionId) return stopSettling();
      if (scrollRuntime.unownedScrollSeq() !== state.unownedScrollSeq) return stopSettling();
      if (scrollRuntime.writeSeq() !== state.writeSeq) return stopSettling();
      const row = findRow(state.position);
      if (!row) return stopSettling();
      if (!placeAtRow(row, state.position.offset)) return stopSettling();
      state.writeSeq = scrollRuntime.writeSeq();
      scrollRuntime.syncJumpButton();
    }

    let settleObserver = null;
    function observeContent() {
      if (settleObserver || !ResizeObserverCtor) return;
      settleObserver = new ResizeObserverCtor(() => reassert());
      settleObserver.observe(content);
    }

    function beginSettle(sessionId, position) {
      if (!ResizeObserverCtor) return;
      observeContent();
      settle = {
        sessionId,
        position,
        sequence,
        writeSeq: scrollRuntime.writeSeq(),
        unownedScrollSeq: scrollRuntime.unownedScrollSeq(),
      };
    }

    function attachAnchor(sessionId, position, row, { detached = false } = {}) {
      // A bottom write queued by the render that just completed would land
      // after this placement and throw the reader to the tail.
      scrollRuntime.cancelScheduledBottomWrites();
      scrollRuntime.disableAutoScroll();
      if (!detached) scrollRuntime.markLiveTail();
      if (!placeAtRow(row, position.offset)) return false;
      scrollRuntime.syncJumpButton();
      beginSettle(sessionId, position);
      return true;
    }

    // Restore the saved position after the render the session is finally
    // showing. Returns true when a position applied (or a materialization was
    // started), false when the session has nothing remembered.
    function restore(sessionId) {
      const id = positionSessionId(sessionId);
      cancel();
      const mySequence = sequence;
      const position = positionFor(id);
      if (!position) return false;
      if (position.kind === "tail") {
        writeTail();
        return true;
      }
      const row = findRow(position);
      if (row) return attachAnchor(id, position, row);
      if (!position.cursor) {
        // Nothing identifies the row any more (older log, trimmed window).
        writeTail();
        return true;
      }
      // The remembered message is not in the freshly rendered tail window: it
      // lived in an older page, which switching away discarded. Materialize the
      // window that contains it — the same rounded window search navigation
      // uses — and place the row there. That window is detached from the live
      // tail, so it keeps its own scroll state and never appends live rows.
      void Promise.resolve()
        .then(() => resolveAnchorWindow(position.cursor))
        .then((loaded) => {
          if (!loaded || mySequence !== sequence || getSelectedSessionId() !== id) return;
          const loadedRow = findRow(position);
          if (!loadedRow) return;
          attachAnchor(id, position, loadedRow, { detached: true });
        })
        .catch((error) => {
          const consoleError = typeof options.consoleError === "function" ? options.consoleError : null;
          if (consoleError) consoleError("transcript position: anchor window failed", error);
        });
      return true;
    }

    return Object.freeze({
      save,
      restore,
      clear,
      cancel,
      positionFor,
      snapshot: () => Object.freeze({
        settling: Boolean(settle),
        settledSessionId: settle ? settle.sessionId : null,
        remembered: positions.size,
      }),
    });
  }

export { createTranscriptPositionController };
