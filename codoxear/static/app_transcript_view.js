(() => {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`transcript view dependency missing: ${name}`);
    return value;
  }

  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`transcript view dependency missing: ${name}`);
    return value;
  }

  function createTranscriptViewController(options = {}) {
    const root = requireObject(options.root, "root");
    const messageRows = requireObject(options.messageRows, "messageRows");
    const transcript = requireObject(options.transcript, "transcript");
    const getMessageRowDeps = requireFunction(options.getMessageRowDeps, "getMessageRowDeps");
    const getSelectedSessionId = requireFunction(options.getSelectedSessionId, "getSelectedSessionId");
    const safeMakeRow = (event, rowOptions) =>
      requireFunction(messageRows.safeMakeRow, "messageRows.safeMakeRow")(event, rowOptions, {
        ...getMessageRowDeps(),
        selectedSessionId: getSelectedSessionId(),
      });

    function renderedMessageRows() {
      return requireFunction(messageRows.renderedMessageRows, "messageRows.renderedMessageRows")(root);
    }

    const renderRuntime = requireFunction(transcript.createTranscriptRenderRuntime, "transcript.createTranscriptRenderRuntime")({
      ...options.renderRuntime,
      document: options.document,
      bottomSentinel: options.bottomSentinel,
      root,
      safeMakeRow,
    });

    return Object.freeze({
      safeMakeRow,
      renderedMessageRows,
      loadedUserMessageRows: () => messageRows.loadedUserMessageRows(root),
      loadedCopyMessageRows: () => messageRows.loadedCopyMessageRows(root),
      rowSearchText: (row) => messageRows.rowSearchText(row),
      clearChatSearchMarks: () => messageRows.clearChatSearchMarks(renderedMessageRows()),
      applyChatSearchMarks: (matches, currentRow, query) => messageRows.applyChatSearchMarks(matches, currentRow, query),
      oldestRenderedHistoryCursor: () => messageRows.oldestRenderedHistoryCursor(renderedMessageRows()),
      firstVisibleMessageRow: (scrollTop) => messageRows.firstVisibleMessageRow(renderedMessageRows(), scrollTop),
      appendEvent: (event) => renderRuntime.appendEvent(event),
      renderTranscript: (events, renderOptions) => renderRuntime.renderTranscript(events, renderOptions),
      renderDetachedTranscriptWindow: (events, renderOptions) => renderRuntime.renderDetachedTranscriptWindow(events, renderOptions),
      prependOlderEvents: (events, renderOptions) => renderRuntime.prependOlderEvents(events, renderOptions),
    });
  }

  window.CodoxearTranscriptView = Object.freeze({ createTranscriptViewController });
})();
