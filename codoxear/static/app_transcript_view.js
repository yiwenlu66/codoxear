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
    const el = requireFunction(options.el, "el");
    const bottomSentinel = requireObject(options.bottomSentinel, "bottomSentinel");
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
      bottomSentinel,
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
      scrollToFirstUnread: (eventId) => renderRuntime.scrollToFirstUnread(eventId),
      renderDetachedTranscriptWindow: (events, renderOptions) => renderRuntime.renderDetachedTranscriptWindow(events, renderOptions),
      prependOlderEvents: (events, renderOptions) => renderRuntime.prependOlderEvents(events, renderOptions),
      renderLoadingRow: () => {
        const row = el("div", { class: "msg-row assistant typing-row transcript-loading-row" });
        row.dataset.role = "assistant";
        row.appendChild(el("div", { class: "msg assistant loading", role: "status", "aria-live": "polite", text: "Loading transcript…" }));
        root.insertBefore(row, bottomSentinel);
      },
      renderLoadErrorRow: ({ message, onRetry }) => {
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
      },
    });
  }

  window.CodoxearTranscriptView = Object.freeze({ createTranscriptViewController });
})();
