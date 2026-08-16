
  const DEFAULT_RETRY_MS = 1000;

  /**
   * Own one transcript EventSource and its retry timer.
   *
   * The caller owns transcript state and declares whether a session/generation
   * is still current. This controller only transports ordered SSE payloads;
   * reconnects always use the caller's latest durable live cursor.
   */
  function createMessageEventSourceController({
    EventSourceImpl = typeof EventSource === "function" ? EventSource : null,
    resolveUrl,
    getSnapshot,
    isActive,
    onOpen,
    onMessage,
    onFallback,
    onStateChange,
    onMalformedMessage = () => {},
    setTimer = setTimeout,
    clearTimer = clearTimeout,
    retryMs = DEFAULT_RETRY_MS,
  }) {
    if (typeof EventSourceImpl !== "function") throw new Error("EventSource is unavailable");
    for (const [name, value] of Object.entries({ resolveUrl, getSnapshot, isActive, onOpen, onMessage, onFallback, onStateChange })) {
      if (typeof value !== "function") throw new Error(`SSE controller requires ${name}`);
    }

    let source = null;
    let retryTimer = null;

    function sourceIsCurrent(candidate, sessionId, generation) {
      return source === candidate && isActive(sessionId, generation);
    }

    function clearRetry() {
      if (retryTimer !== null) clearTimer(retryTimer);
      retryTimer = null;
    }

    function close() {
      clearRetry();
      const current = source;
      source = null;
      onStateChange(false);
      if (current && typeof current.close === "function") {
        try { current.close(); } catch (_error) {}
      }
    }

    function scheduleRetry(sessionId, generation) {
      if (retryTimer !== null || !isActive(sessionId, generation)) return;
      retryTimer = setTimer(() => {
        retryTimer = null;
        open(sessionId, generation);
      }, Math.max(0, Number(retryMs) || DEFAULT_RETRY_MS));
    }

    function fail(candidate, sessionId, generation) {
      if (!sourceIsCurrent(candidate, sessionId, generation)) return;
      if (typeof candidate.close === "function") {
        try { candidate.close(); } catch (_error) {}
      }
      source = null;
      onStateChange(false);
      onFallback(sessionId, generation);
      scheduleRetry(sessionId, generation);
    }

    function open(sessionId, generation) {
      if (!isActive(sessionId, generation)) return false;
      const snapshot = getSnapshot();
      if (!snapshot || snapshot.state !== "bound" || !snapshot.liveCursor) return false;
      close();
      const candidate = new EventSourceImpl(resolveUrl(`/api/sessions/${sessionId}/live?cursor=${encodeURIComponent(snapshot.liveCursor)}`));
      source = candidate;
      candidate.onopen = () => {
        if (!sourceIsCurrent(candidate, sessionId, generation)) return;
        onStateChange(true);
        onOpen(sessionId, generation);
      };
      candidate.addEventListener("message", (event) => {
        if (!sourceIsCurrent(candidate, sessionId, generation)) return;
        let data;
        try {
          data = JSON.parse(event.data);
        } catch (error) {
          onMalformedMessage(error);
          return;
        }
        Promise.resolve(onMessage(sessionId, generation, data)).catch(() => fail(candidate, sessionId, generation));
      });
      candidate.addEventListener("error", () => fail(candidate, sessionId, generation));
      return true;
    }

    function resume(sessionId, generation) {
      if (!isActive(sessionId, generation) || source) return false;
      clearRetry();
      return open(sessionId, generation);
    }

    return Object.freeze({ close, open, resume });
  }

export { DEFAULT_RETRY_MS, createMessageEventSourceController };
