import * as CodoxearPolling from "./app_polling.js";
import * as CodoxearTranscript from "./app_transcript.js";


// Message data-flow authority. Owns confirmed sends, initial-tail/poll request
  // cancellation, SSE connection/retry state, polling cadence/backoff, the
  // shared SSE/poll live-delta reducer, and typing-count reconciliation. Session
  // selection itself remains in app.js; selected id + generation are injected
  // so every asynchronous result is rejected after a selection change.



  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`message flow dependency missing: ${name}`);
    return value;
  }

  const CONTROL_SLASH_COMMANDS = new Set(["model", "effort", "thinking", "new"]);

  function slashCommandName(raw) {
    const match = String(raw || "").trim().match(/^\/([^\s/]+)/);
    return match ? match[1].toLowerCase() : "";
  }

  function isKnownControlSlashCommand(raw, session) {
    const command = slashCommandName(raw);
    if (!command) return false;
    if (CONTROL_SLASH_COMMANDS.has(command)) return true;
    return Array.isArray(session && session.slash_commands)
      && session.slash_commands.some((entry) => String(entry && entry.name || "").replace(/^\//, "").toLowerCase() === command);
  }

  function isModelControlCommand(raw) {
    return slashCommandName(raw) === "model";
  }

  function createMessageFlowController(options = {}) {
    if (!options || typeof options !== "object") throw new TypeError("message flow dependency missing: options");

    const getGeneration = requireFunction(options.getGeneration, "getGeneration");
    const isAppDisposed = requireFunction(options.isAppDisposed, "isAppDisposed");
    const sessionState = options.sessionState;
    if (!sessionState || typeof sessionState.get !== "function" || typeof sessionState.set !== "function" || typeof sessionState.applyRuntime !== "function") {
      throw new TypeError("message flow dependency missing: sessionState");
    }
    const getSessionInfo = requireFunction(options.getSessionInfo, "getSessionInfo");
    const patchSessionInfo = requireFunction(options.patchSessionInfo, "patchSessionInfo");
    const sessionLaunchFailed = requireFunction(options.sessionLaunchFailed, "sessionLaunchFailed");
    const api = requireFunction(options.api, "api");
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const handleAppAuthLoss = requireFunction(options.handleAppAuthLoss, "handleAppAuthLoss");
    const refreshSessions = requireFunction(options.refreshSessions, "refreshSessions");
    const openSession = requireFunction(options.openSession, "openSession");
    const clearSelectedSessionAfterRemoval = requireFunction(options.clearSelectedSessionAfterRemoval, "clearSelectedSessionAfterRemoval");
    const activeTranscriptSnapshot = requireFunction(options.activeTranscriptSnapshot, "activeTranscriptSnapshot");
    const updateSessionTranscriptSlot = requireFunction(options.updateSessionTranscriptSlot, "updateSessionTranscriptSlot");
    const renderPendingTranscriptSlot = requireFunction(options.renderPendingTranscriptSlot, "renderPendingTranscriptSlot");
    const renderSessionTail = requireFunction(options.renderSessionTail, "renderSessionTail");
    const applySessionRuntimeFromTail = requireFunction(options.applySessionRuntimeFromTail, "applySessionRuntimeFromTail");
    const resetChatRenderState = requireFunction(options.resetChatRenderState, "resetChatRenderState");
    const setAttachCount = requireFunction(options.setAttachCount, "setAttachCount");
    const setLiveCursor = requireFunction(options.setLiveCursor, "setLiveCursor");
    const appendEvents = typeof options.appendEvents === "function"
      ? options.appendEvents
      : (events) => {
        const appendEvent = requireFunction(options.appendEvent, "appendEvents");
        let changed = false;
        for (const event of Array.isArray(events) ? events : []) changed = appendEvent(event) || changed;
        return changed;
      };
    const appendTailSnapshotEvents = requireFunction(options.appendTailSnapshotEvents, "appendTailSnapshotEvents");
    const updateSessionTitle = requireFunction(options.updateSessionTitle, "updateSessionTitle");
    const initPageLimit = requireFunction(options.initPageLimit, "initPageLimit");
    const typingRowRuntime = options.typingRowRuntime;
    if (
      !typingRowRuntime ||
      typeof typingRowRuntime.snapshot !== "function" ||
      typeof typingRowRuntime.updateTypingStats !== "function" ||
      typeof typingRowRuntime.updateSubagentGauge !== "function" ||
      typeof typingRowRuntime.resetTypingStats !== "function"
    )
      throw new TypeError("message flow dependency missing: typingRowRuntime");

    // Confirmed-send effects. The composer owns input/modal UI only and calls
    // this controller at the send boundary.
    const getStagedAttachments = requireFunction(options.getStagedAttachments, "getStagedAttachments");
    const normalizedStagedAttachments = requireFunction(options.normalizedStagedAttachments, "normalizedStagedAttachments");
    const setSelectedSessionPendingAttachment = requireFunction(options.setSelectedSessionPendingAttachment, "setSelectedSessionPendingAttachment");
    const syncSendButtonState = requireFunction(options.syncSendButtonState, "syncSendButtonState");
    const syncAttachButtonState = requireFunction(options.syncAttachButtonState, "syncAttachButtonState");
    const syncQueueSubmitState = requireFunction(options.syncQueueSubmitState, "syncQueueSubmitState");
    const syncRecoveryUiForSession = requireFunction(options.syncRecoveryUiForSession, "syncRecoveryUiForSession");
    const confirmAction = requireFunction(options.confirmAction, "confirmAction");
    const setToast = requireFunction(options.setToast, "setToast");
    const isTranscriptRenewalCommand = requireFunction(options.isTranscriptRenewalCommand, "isTranscriptRenewalCommand");
    const nextLocalEchoId = requireFunction(options.nextLocalEchoId, "nextLocalEchoId");
    const renderedAtLiveTail = requireFunction(options.renderedAtLiveTail, "renderedAtLiveTail");
    const getSessionTranscriptSlot = requireFunction(options.getSessionTranscriptSlot, "getSessionTranscriptSlot");
    const addPendingUser = requireFunction(options.addPendingUser, "addPendingUser");
    const deleteTailCache = requireFunction(options.deleteTailCache, "deleteTailCache");
    const beginTranscriptRenewal = requireFunction(options.beginTranscriptRenewal, "beginTranscriptRenewal");
    const clearLiveCursor = requireFunction(options.clearLiveCursor, "clearLiveCursor");
    const invalidateOlderLoad = requireFunction(options.invalidateOlderLoad, "invalidateOlderLoad");
    const dropPendingUser = requireFunction(options.dropPendingUser, "dropPendingUser");
    const removePendingUserRow = requireFunction(options.removePendingUserRow, "removePendingUserRow");
    const hasPendingForSession = requireFunction(options.hasPendingForSession, "hasPendingForSession");

    const visibilityState = typeof options.visibilityState === "function" ? options.visibilityState : () => document.visibilityState;
    const navigatorValue = typeof options.navigatorValue === "function" ? options.navigatorValue : () => (typeof navigator === "undefined" ? undefined : navigator);
    const reportTransportSuccess = typeof options.reportTransportSuccess === "function" ? options.reportTransportSuccess : () => {};
    const reportTransportFailure = typeof options.reportTransportFailure === "function" ? options.reportTransportFailure : () => {};
    const EventSourceCtor = Object.prototype.hasOwnProperty.call(options, "EventSource") ? options.EventSource : window.EventSource;
    const AbortControllerCtor = Object.prototype.hasOwnProperty.call(options, "AbortController") ? options.AbortController : window.AbortController;
    const setTimeoutFn = typeof options.setTimeout === "function" ? options.setTimeout : window.setTimeout.bind(window);
    const clearTimeoutFn = typeof options.clearTimeout === "function" ? options.clearTimeout : window.clearTimeout.bind(window);
    const now = typeof options.now === "function" ? options.now : () => Date.now();
    const consoleWarn = typeof options.consoleWarn === "function" ? options.consoleWarn : console.warn.bind(console);
    const consoleError = typeof options.consoleError === "function" ? options.consoleError : console.error.bind(console);

    let openSessionTailAbortController = null;
    let messagePollAbortController = null;
    let messageEventSource = null;
    let messageSseRetryTimer = null;
    let messageSseOpen = false;
    let messageSseFallbackUntil = 0;
    let pollTimer = null;
    let pollLoopBusy = false;
    let pollKickPending = false;
    let pollKickDelayMs = null;
    let messagePollErrorStreak = 0;
    let messageTransportUnavailable = false;
    let pollFastUntilMs = 0;

    function isCurrent(sessionId, generation) {
      return !isAppDisposed() && sessionState.get("selected") === sessionId && getGeneration() === generation;
    }

    function abortController(controller) {
      if (!controller || typeof controller.abort !== "function") return;
      try { controller.abort(); } catch (_error) {}
    }

    function abortOpenSessionTailRequest() {
      const controller = openSessionTailAbortController;
      openSessionTailAbortController = null;
      abortController(controller);
    }

    function beginOpenSessionTailRequest(sessionId, generation) {
      abortOpenSessionTailRequest();
      const controller = typeof AbortControllerCtor === "function" ? new AbortControllerCtor() : null;
      openSessionTailAbortController = controller;
      return Object.freeze({ sessionId, generation, controller, signal: controller ? controller.signal : undefined });
    }

    function isCurrentOpenSessionTailRequest(request) {
      return Boolean(request && sessionState.get("selected") === request.sessionId && getGeneration() === request.generation);
    }

    function isOpenSessionTailAbortError(request, error) {
      return Boolean(error && error.name === "AbortError" && request && request.signal && request.signal.aborted);
    }

    function finishOpenSessionTailRequest(request) {
      if (request && openSessionTailAbortController === request.controller) openSessionTailAbortController = null;
    }

    function abortMessagePollRequest() {
      const controller = messagePollAbortController;
      messagePollAbortController = null;
      abortController(controller);
    }

    function beginMessagePollRequest(sessionId, generation) {
      abortMessagePollRequest();
      const controller = typeof AbortControllerCtor === "function" ? new AbortControllerCtor() : null;
      messagePollAbortController = controller;
      return Object.freeze({ sessionId, generation, controller, signal: controller ? controller.signal : undefined });
    }

    function isMessagePollAbortError(request, error) {
      return Boolean(error && error.name === "AbortError" && request && request.signal && request.signal.aborted);
    }

    function finishMessagePollRequest(request) {
      if (request && messagePollAbortController === request.controller) messagePollAbortController = null;
    }

    function browserOffline() {
      return CodoxearPolling.browserOffline(navigatorValue());
    }

    function messagePollDelayMs(at = now()) {
      return CodoxearPolling.messagePollDelayMs({
        now: at,
        visibilityState: visibilityState(),
        offline: browserOffline(),
        errorStreak: messagePollErrorStreak,
        pollFastUntilMs,
        turnOpen: sessionState.get("turnOpen"),
      });
    }

    function markMessagePollSuccess() {
      messagePollErrorStreak = 0;
      messageTransportUnavailable = false;
      reportTransportSuccess();
    }

    function markMessagePollFailure(transportFailed = true) {
      messagePollErrorStreak = Math.min(messagePollErrorStreak + 1, 20);
      if (!transportFailed) return;
      messageTransportUnavailable = true;
      reportTransportFailure();
      closeMessageEventSource();
    }

    function resetMessagePollBackoff() {
      messagePollErrorStreak = 0;
    }

    function normalizeMessagePollKickDelay(ms = 0) {
      return CodoxearPolling.normalizeMessagePollKickDelay({
        requested: ms,
        visibilityState: visibilityState(),
        offline: browserOffline(),
        errorStreak: messagePollErrorStreak,
        pollFastUntilMs,
        turnOpen: sessionState.get("turnOpen"),
      });
    }

    function closeMessageEventSource() {
      if (messageSseRetryTimer) clearTimeoutFn(messageSseRetryTimer);
      messageSseRetryTimer = null;
      const source = messageEventSource;
      messageEventSource = null;
      messageSseOpen = false;
      if (source && typeof source.close === "function") {
        try { source.close(); } catch (_error) {}
      }
    }

    function scheduleMessageEventSourceRetry(sessionId, generation) {
      if (messageTransportUnavailable || !isCurrent(sessionId, generation) || messageSseRetryTimer || visibilityState() !== "visible") return;
      const delay = Math.max(1000, messageSseFallbackUntil - now());
      messageSseRetryTimer = setTimeoutFn(() => {
        messageSseRetryTimer = null;
        if (isCurrent(sessionId, generation) && visibilityState() === "visible") openMessageEventSource(sessionId, generation);
      }, delay);
    }

    function openMessageEventSource(sessionId = sessionState.get("selected"), generation = getGeneration()) {
      if (!sessionId || !isCurrent(sessionId, generation) || visibilityState() !== "visible" || typeof EventSourceCtor !== "function") return false;
      const snapshot = activeTranscriptSnapshot();
      if (snapshot.state !== "bound" || !snapshot.liveCursor) return false;
      // A visibility resume can occur before EventSource calls `open`. Reuse
      // that in-flight connection instead of starting both a second stream and
      // a fallback HTTP poll for the same cursor.
      if (messageEventSource) return true;
      const url = resolveAppUrl(`/api/sessions/${sessionId}/live?cursor=${encodeURIComponent(snapshot.liveCursor)}`);
      const source = new EventSourceCtor(url);
      messageEventSource = source;
      source.onopen = () => {
        if (messageEventSource !== source || !isCurrent(sessionId, generation)) return;
        messageSseOpen = true;
        messageSseFallbackUntil = 0;
        markMessagePollSuccess();
        abortMessagePollRequest();
        if (pollTimer) clearTimeoutFn(pollTimer);
        pollTimer = null;
      };
      source.addEventListener("message", (event) => {
        if (messageEventSource !== source || !isCurrent(sessionId, generation)) return;
        let data;
        try { data = JSON.parse(event.data); } catch (error) {
          consoleWarn("message SSE payload was invalid", error);
          return;
        }
        Promise.resolve(applyLiveMessageData(sessionId, generation, data)).catch((error) => {
          consoleWarn("message SSE update failed", error);
          source.close();
          if (messageEventSource === source) {
            messageEventSource = null;
            messageSseOpen = false;
            messageSseFallbackUntil = now() + 6000;
            kickPoll(0);
            scheduleMessageEventSourceRetry(sessionId, generation);
          }
        });
      });
      source.addEventListener("error", () => {
        if (messageEventSource !== source || !isCurrent(sessionId, generation)) return;
        source.close();
        messageEventSource = null;
        messageSseOpen = false;
        messageSseFallbackUntil = now() + 6000;
        markMessagePollFailure(false);
        kickPoll(0);
        scheduleMessageEventSourceRetry(sessionId, generation);
      });
      return true;
    }

    function resumeLiveDelivery() {
      if (isAppDisposed() || visibilityState() !== "visible" || !sessionState.get("selected")) return;
      if (!messageSseOpen && !openMessageEventSource()) kickPoll(0);
    }

    function clearPollSchedule() {
      if (pollTimer) clearTimeoutFn(pollTimer);
      pollTimer = null;
      pollKickPending = false;
      pollKickDelayMs = null;
    }

    function prepareSessionOpen() {
      closeMessageEventSource();
      abortOpenSessionTailRequest();
      abortMessagePollRequest();
      clearPollSchedule();
    }

    function stop() {
      closeMessageEventSource();
      abortOpenSessionTailRequest();
      abortMessagePollRequest();
      clearPollSchedule();
      messagePollErrorStreak = 0;
      messageTransportUnavailable = false;
      pollFastUntilMs = 0;
    }

    function updateTypingStatsFromSession(session, { updateRuntime = true, updateSubagents = true } = {}) {
      // updateSubagents is the pre-store-migration suppression name still used
      // by tail/cache callers that apply their payload runtime separately.
      const shouldApplyRuntime = updateRuntime && updateSubagents;
      const running = Boolean(session && session.busy);
      if (shouldApplyRuntime && running) sessionState.set("turnOpen", true);

      if (session) {
        const thinkingMode = CodoxearTranscript.thinkingModeForTokens(session.thinking_tokens);
        const stats = {
          thinking: session.thinking,
          thinkingTokens: session.thinking_tokens,
          thinkingMode,
          tools: session.tools,
        };
        // Live deltas are exact; session-list snapshots are resumable but can
        // lag. While the turn is open a snapshot may raise counters but never
        // lower exact values already observed from the live feed.
        if (!sessionState.get("turnOpen")) {
          typingRowRuntime.updateTypingStats(stats);
        } else {
          const current = typingRowRuntime.snapshot().stats || { thinking: 0, thinkingTokens: 0, tools: 0 };
          typingRowRuntime.updateTypingStats({
            thinking: Math.max(current.thinking, stats.thinking || 0),
            thinkingTokens: Math.max(current.thinkingTokens, stats.thinkingTokens || 0),
            thinkingMode,
            tools: Math.max(current.tools, stats.tools || 0),
          });
        }
      }

      if (shouldApplyRuntime) {
        const queueLen = session && Number.isFinite(Number(session.queue_len)) ? Number(session.queue_len) : 0;
        const subagentsRunning = session ? Math.max(0, Math.floor(Number(session.subagents_running) || 0)) : 0;
        sessionState.set("turnOpen", running);
        sessionState.applyRuntime({ running, queueLen, token: session ? session.token || null : null, subagentsRunning });
      }
    }

    function applyTypingMetaDelta(data) {
      const delta = data && data.meta_delta;
      if (!delta || typeof delta !== "object") return;
      const currentStats = typingRowRuntime.snapshot().stats || { thinkingTokens: 0 };
      const thinkingMode = CodoxearTranscript.thinkingModeForTokens(
        Math.max(Number(currentStats.thinkingTokens) || 0, Number(delta.thinking_tokens) || 0),
      );
      typingRowRuntime.updateTypingStats(
        {
          thinking: delta.thinking,
          thinkingTokens: delta.thinking_tokens,
          thinkingMode,
          tools: delta.tool,
        },
        { delta: true },
      );
    }

    async function applyLiveMessageData(sessionId, generation, data) {
      if (!isCurrent(sessionId, generation)) return;
      markMessagePollSuccess();
      const slotInfo = CodoxearTranscript.transcriptSnapshotFromData(data);
      const nowBusy = Boolean(data.busy);
      const wasTurnOpen = sessionState.get("turnOpen");
      const active = activeTranscriptSnapshot();
      if (active.state === "bound" && slotInfo.state === "pending_bind") {
        // Server is re-binding the log, but our existing messages are still
        // valid. Do NOT clear the DOM — just update the slot metadata and
        // let the next poll/SSE append events on top of what's visible.
        updateSessionTranscriptSlot(sessionId, data);
        applySessionRuntimeFromTail(sessionId, data);
        return;
      }
      if (active.state === "bound" && slotInfo.state === "bound" && slotInfo.logPath !== active.logPath) {
        // Log path changed — but don't clear the visible transcript until
        // we have replacement content. If the reload fails, the user keeps
        // seeing their existing messages instead of a blank panel.
        try {
          await openSession(sessionId, { useCache: false });
        } catch (e) {
          if (e && e.status === 401) handleAppAuthLoss();
          else console.error("log path change reload failed, keeping existing transcript", e);
        }
        return;
      }
      const nextLiveCursor = typeof data.live_cursor === "string" && data.live_cursor ? data.live_cursor : null;
      setLiveCursor(nextLiveCursor);
      const events = Array.isArray(data.events) ? data.events : [];
      appendEvents(events);
      const turnStart = Boolean(data.turn_start);
      const turnEnd = Boolean(data.turn_end);
      const turnAborted = Boolean(data.turn_aborted);
      const newTurn = CodoxearTranscript.startsTypingCountWindow({ wasTurnOpen, turnStart, nowBusy });
      if (newTurn && CodoxearTranscript.hasHumanOriginatedUserEvent(events)) typingRowRuntime.resetTypingStats();
      let turnOpen = wasTurnOpen;
      if (turnStart) turnOpen = true;
      if (!turnOpen && nowBusy) turnOpen = true;
      if ((turnEnd || turnAborted) && turnOpen) turnOpen = false;
      if (turnOpen && !nowBusy) turnOpen = false;
      sessionState.set("turnOpen", turnOpen);
      applyTypingMetaDelta(data);
      const running = Boolean(turnOpen || nowBusy);
      const queueLen = Number.isFinite(Number(data.queue_len)) ? Number(data.queue_len) : 0;
      sessionState.applyRuntime({ running, queueLen, token: data.token || null });
      const session = getSessionInfo(sessionId);
      if (events.length) {
        appendTailSnapshotEvents(sessionId, events, {
          session,
          liveCursor: nextLiveCursor,
          busy: running,
          queueLen: data.queue_len,
          token: data.token,
          identityData: data,
        });
      }
      if (session) updateSessionTitle(session);
    }

    async function pollMessages(sessionId = sessionState.get("selected"), generation = getGeneration()) {
      if (isAppDisposed() || !sessionId) return;
      const reconnectSseAfterSuccess = messageTransportUnavailable;
      let pollRequest = null;
      try {
        const active = activeTranscriptSnapshot();
        if (!active.liveCursor) {
          if (active.state === "pending_bind") {
            pollRequest = beginMessagePollRequest(sessionId, generation);
            const data = await api(`/api/sessions/${sessionId}/messages/tail?limit=${initPageLimit()}`, { signal: pollRequest.signal });
            if (!isCurrent(sessionId, generation)) return;
            markMessagePollSuccess();
            const slotChange = updateSessionTranscriptSlot(sessionId, data);
            if (slotChange.ignoredStaleBound) {
              applySessionRuntimeFromTail(sessionId, { transcript_state: "pending_bind", busy: data.busy, queue_len: data.queue_len, token: data.token });
              return;
            }
            // Polling must NEVER clear and re-render the transcript.
            // Append any events we don't already have (dedup handles
            // duplicates) and set the liveCursor for incremental polling.
            // Full re-render only happens in openSession (initial load).
            applySessionRuntimeFromTail(sessionId, data);
            const tailEvents = Array.isArray(data.events) ? data.events : [];
            appendEvents(tailEvents);
            if (reconnectSseAfterSuccess) resumeLiveDelivery();
            return;
          }
          if (active.state === "failed") return;
          // Bound but no cursor (rare): fetch tail, append events, set
          // cursor. Same no-clear policy as pending_bind above.
          pollRequest = beginMessagePollRequest(sessionId, generation);
          const boundData = await api(`/api/sessions/${sessionId}/messages/tail?limit=${initPageLimit()}`, { signal: pollRequest.signal });
          if (!isCurrent(sessionId, generation)) return;
          markMessagePollSuccess();
          updateSessionTranscriptSlot(sessionId, boundData);
          applySessionRuntimeFromTail(sessionId, boundData);
          const boundEvents = Array.isArray(boundData.events) ? boundData.events : [];
          appendEvents(boundEvents);
          if (reconnectSseAfterSuccess) resumeLiveDelivery();
          return;
        }
        const requestedCursor = active.liveCursor;
        pollRequest = beginMessagePollRequest(sessionId, generation);
        const data = await api(`/api/sessions/${sessionId}/messages/live?cursor=${encodeURIComponent(requestedCursor)}`, { signal: pollRequest.signal });
        await applyLiveMessageData(sessionId, generation, data);
        if (reconnectSseAfterSuccess) resumeLiveDelivery();
      } catch (error) {
        if (error && error.status === 401) {
          handleAppAuthLoss();
          return;
        }
        if (isMessagePollAbortError(pollRequest, error)) return;
        if (!isCurrent(sessionId, generation)) return;
        if (error && error.status === 409) {
          await openSession(sessionId, { useCache: false });
          return;
        }
        if (error && error.status === 404) {
          clearSelectedSessionAfterRemoval(sessionId, { incrementPollGen: true, clearPollState: true });
          try {
            await refreshSessions();
          } catch (refreshError) {
            consoleError("refreshSessions failed after session disappeared", refreshError);
            setToast(`refresh error: ${refreshError && refreshError.message ? refreshError.message : "unknown error"}`);
          }
          return;
        }
        markMessagePollFailure(!(error && typeof error.status === "number"));
        if (error && typeof error.status === "number") setToast(`error: ${error.message}`);
        else consoleWarn("message poll network error", error && error.message);
      } finally {
        finishMessagePollRequest(pollRequest);
      }
    }

    async function pollLoop() {
      if (isAppDisposed() || !sessionState.get("selected") || messageSseOpen) return;
      if (pollLoopBusy) {
        pollKickPending = true;
        return;
      }
      pollLoopBusy = true;
      const sessionId = sessionState.get("selected");
      const generation = getGeneration();
      try {
        await pollMessages(sessionId, generation);
      } finally {
        pollLoopBusy = false;
      }
      if (pollKickPending) {
        const delay = pollKickDelayMs == null ? 0 : pollKickDelayMs;
        pollKickPending = false;
        pollKickDelayMs = null;
        kickPoll(delay);
        return;
      }
      if (!isCurrent(sessionId, generation)) return;
      pollTimer = setTimeoutFn(pollLoop, messagePollDelayMs());
    }

    function kickPoll(ms = 0) {
      if (isAppDisposed() || messageSseOpen) return;
      const delay = normalizeMessagePollKickDelay(ms);
      if (pollTimer) {
        clearTimeoutFn(pollTimer);
        pollTimer = null;
      }
      if (pollLoopBusy) {
        pollKickPending = true;
        pollKickDelayMs = delay;
        return;
      }
      pollTimer = setTimeoutFn(pollLoop, delay);
    }

    function setPollFastUntilMs(value) {
      pollFastUntilMs = Number(value) || 0;
    }

    async function sendText(raw, { sid = null } = {}) {
      const sessionId = sid || sessionState.get("selected");
      if (!sessionId || !raw || !raw.trim() || sessionState.get("sending")) return false;
      const renderHere = sessionId === sessionState.get("selected");
      const renewsTranscript = isTranscriptRenewalCommand(raw, sessionId);
      const sessionInfo = getSessionInfo(sessionId) || null;
      const isControlSlashCommand = isKnownControlSlashCommand(raw, sessionInfo);
      const modelControlCommand = isModelControlCommand(raw);
      if (sessionInfo && sessionLaunchFailed(sessionInfo)) {
        setToast("failed launch cannot receive messages");
        return false;
      }
      const stagedAttachments = getStagedAttachments();
      const localAttachmentCount = renderHere ? stagedAttachments.length : normalizedStagedAttachments(sessionInfo && sessionInfo.staged_attachments).length;
      let allowPendingAttachment = localAttachmentCount > 0;
      if (!allowPendingAttachment && sessionInfo && sessionInfo.pending_attachment) {
        const confirmed = await confirmAction({
          title: "Send pending attachment?",
          message: "This session has a pending file attachment. Send it with this message?",
          confirmText: "Send with attachment",
          cancelText: "Cancel",
        });
        if (!confirmed) return false;
        allowPendingAttachment = true;
      }
      const continuesOpenTurn = renderHere && sessionState.get("running");
      sessionState.set("sending", true);
      setToast("sending...");

      const localId = nextLocalEchoId();
      const startedAt = now() / 1000;
      if (renderHere && !continuesOpenTurn && !isControlSlashCommand) typingRowRuntime.resetTypingStats();
      if (renderHere && !renewsTranscript && !isControlSlashCommand) {
        // NEVER clear the transcript on send. Just scroll to bottom
        // so the user sees the new message and the response.
        const slot = getSessionTranscriptSlot(sessionId);
        addPendingUser({ id: localId, sessionId, epoch: slot.epoch, text: raw, t0: startedAt });
        appendEvents([{ role: "user", text: raw, pending: true, localId, ts: startedAt }]);
        sessionState.set("turnOpen", true);
        sessionState.applyRuntime({ running: true });
      }
      try {
        const response = await api(`/api/sessions/${sessionId}/send`, { method: "POST", body: { text: raw, allow_pending_attachment: allowPendingAttachment } });
        if (renderHere && renewsTranscript) {
          deleteTailCache(sessionId);
          beginTranscriptRenewal(sessionId);
          // A send may advance the transcript epoch, but it must never clear
          // or replace the currently visible conversation. The next bound tail
          // supplies the explicit replacement boundary if one is needed.
          clearLiveCursor();
          invalidateOlderLoad();
        }
        const attachmentCleanupError = response && (response.attachment_cleanup_error || response.attachments_cleanup_error)
          ? String(response.attachment_cleanup_error || response.attachments_cleanup_error)
          : "";
        const sendStateCleanupError = response && response.send_state_cleanup_error ? String(response.send_state_cleanup_error) : "";
        const deliveredToast = response.queued ? `queued (queue ${response.queue_len})` : "sent";
        const cleanupWarnings = [];
        if (attachmentCleanupError) cleanupWarnings.push(`attachment cleanup failed: ${attachmentCleanupError}`);
        if (sendStateCleanupError) cleanupWarnings.push(`send state cleanup failed: ${sendStateCleanupError}`);
        setToast(cleanupWarnings.length ? `${deliveredToast}; ${cleanupWarnings.join("; ")}` : deliveredToast);
        if (allowPendingAttachment && !attachmentCleanupError) {
          setSelectedSessionPendingAttachment(sessionId, false);
          setAttachCount(0);
        }
        setPollFastUntilMs(now() + 5000);
        kickPoll(0);
        void refreshSessions().catch((error) => {
          if (error && error.status === 401) handleAppAuthLoss();
          else consoleError("refreshSessions failed", error);
        });
        if (modelControlCommand) {
          setTimeoutFn(() => {
            void refreshSessions().catch((error) => {
              if (error && error.status === 401) handleAppAuthLoss();
              else consoleError("refreshSessions failed after model change", error);
            });
          }, 1500);
        }
        return true;
      } catch (error) {
        if (error && error.status === 401) {
          handleAppAuthLoss();
          return false;
        }
        const commitUnknown = Boolean(error && error.obj && error.obj.commit_unknown);
        if (commitUnknown) {
          setToast("send status unknown; check transcript before retrying");
          patchSessionInfo(sessionId, {
            commit_unknown_send: true,
            commit_unknown_send_text: raw,
            commit_unknown_send_ts: now() / 1000,
          });
          syncSendButtonState();
          syncQueueSubmitState();
          syncAttachButtonState();
          setPollFastUntilMs(now() + 4000);
          kickPoll(0);
          void refreshSessions().catch((refreshError) => {
            if (refreshError && refreshError.status === 401) handleAppAuthLoss();
            else consoleError("refreshSessions failed", refreshError);
          });
        } else {
          setToast(`send error: ${error && error.message ? error.message : "unknown error"}`);
        }
        if (!commitUnknown && sessionInfo && sessionInfo.pending_attachment && /broker must be restarted/i.test(String(error && error.message ? error.message : ""))) {
          const clearPending = await confirmAction({
            title: "Clear pending attachment state?",
            message: "This session has a pending attachment but the current broker cannot confirm sends. Clear the browser pending-attachment state only if you already handled it in the terminal?",
            confirmText: "Clear state",
            cancelText: "Cancel",
            destructive: true,
          });
          if (clearPending) {
            try {
              await api(`/api/sessions/${sessionId}/pending_attachment/clear`, { method: "POST", body: {} });
              setToast("pending attachment state cleared");
              if (sessionState.get("selected") === sessionId) setSelectedSessionPendingAttachment(sessionId, false);
              void refreshSessions().catch((refreshError) => {
                if (refreshError && refreshError.status === 401) handleAppAuthLoss();
                else consoleError("refreshSessions failed", refreshError);
              });
            } catch (clearError) {
              if (clearError && clearError.status === 401) {
                handleAppAuthLoss();
                return false;
              }
              setToast(`clear pending attachment error: ${clearError && clearError.message ? clearError.message : "unknown error"}`);
            }
          }
        }
        if (renderHere) {
          dropPendingUser(sessionId, localId);
          removePendingUserRow(localId);
          if (!hasPendingForSession(sessionId)) {
            sessionState.set("turnOpen", false);
            sessionState.applyRuntime({ running: false });
          }
          if (commitUnknown) syncRecoveryUiForSession(sessionId);
        }
        return false;
      } finally {
        sessionState.set("sending", false);
      }
    }

    return Object.freeze({
      abortMessagePollRequest,
      abortOpenSessionTailRequest,
      applyLiveMessageData,
      beginOpenSessionTailRequest,
      clearPollSchedule,
      closeMessageEventSource,
      finishOpenSessionTailRequest,
      isCurrentOpenSessionTailRequest,
      isOpenSessionTailAbortError,
      kickPoll,
      markMessagePollFailure,
      markMessagePollSuccess,
      messagePollDelayMs,
      openMessageEventSource,
      pollMessages,
      prepareSessionOpen,
      resetMessagePollBackoff,
      resumeLiveDelivery,
      sendText,
      setPollFastUntilMs,
      stop,
      updateTypingStatsFromSession,
      snapshot: () => Object.freeze({
        messageSseOpen,
        messagePollErrorStreak,
        messageTransportUnavailable,
        pollFastUntilMs,
        hasEventSource: Boolean(messageEventSource),
        hasRetryTimer: Boolean(messageSseRetryTimer),
      }),
    });
  }

export { createMessageFlowController };
