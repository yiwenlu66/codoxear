
  const POLLING_INTERVALS = Object.freeze({
    SESSION_POLL_VISIBLE_MS: 5000,
    SESSION_POLL_HIDDEN_MS: 15000,
    SECONDARY_POLL_VISIBLE_MS: 30000,
    SECONDARY_POLL_HIDDEN_MS: 60000,
    MESSAGE_POLL_FAST_MS: 300,
    MESSAGE_POLL_RUNNING_MS: 500,
    MESSAGE_POLL_IDLE_MS: 1500,
    MESSAGE_POLL_HIDDEN_MS: 5000,
    MESSAGE_POLL_OFFLINE_MS: 15000,
    MESSAGE_POLL_ERROR_MIN_MS: 2000,
    MESSAGE_POLL_ERROR_MAX_MS: 30000,
  });

  function visibilityPollDelayMs(visibilityState, visibleMs, hiddenMs) {
    return visibilityState === "hidden" ? hiddenMs : visibleMs;
  }

  // The async epoch invalidates work across session selection, transcript,
  // history, search, and live delivery. It has no scheduling authority.
  function createAsyncEpoch() {
    let generation = 0;

    function nextGeneration() {
      generation += 1;
      return generation;
    }

    function incrementGeneration() {
      generation += 1;
    }

    return Object.freeze({
      currentGeneration: () => generation,
      nextGeneration,
      incrementGeneration,
    });
  }

  // Polling owns timer handles, retry streaks, and enabled state. Composition
  // supplies each loop's tick because its work belongs to application
  // controllers, not the scheduling mechanism.
  function createPollingRuntime(options = {}) {
    function requirePollingFunction(value, name) {
      if (typeof value !== "function") throw new TypeError(`polling runtime dependency missing: ${name}`);
      return value;
    }

    const scheduleTimer = requirePollingFunction(options.setTimeout, "setTimeout");
    const cancelTimer = requirePollingFunction(options.clearTimeout, "clearTimeout");
    let sessionsTimer = null;
    let secondaryTimer = null;
    let sessionsPollingEnabled = true;
    let secondaryPollingEnabled = true;
    let sessionsPollErrorStreak = 0;
    let secondaryPollErrorStreak = 0;

    function cancelSessions() {
      if (sessionsTimer) cancelTimer(sessionsTimer);
      sessionsTimer = null;
    }

    function cancelSecondary() {
      if (secondaryTimer) cancelTimer(secondaryTimer);
      secondaryTimer = null;
    }

    function cancelAll() {
      cancelSessions();
      cancelSecondary();
    }

    function scheduleSessions(delayMs, runTick) {
      if (!sessionsPollingEnabled) return;
      const tick = requirePollingFunction(runTick, "runSessionsTick");
      cancelSessions();
      sessionsTimer = scheduleTimer(() => {
        sessionsTimer = null;
        if (sessionsPollingEnabled) void tick();
      }, Math.max(0, Number(delayMs) || 0));
    }

    function scheduleSecondary(delayMs, runTick) {
      if (!secondaryPollingEnabled) return;
      const tick = requirePollingFunction(runTick, "runSecondaryTick");
      cancelSecondary();
      secondaryTimer = scheduleTimer(() => {
        secondaryTimer = null;
        if (secondaryPollingEnabled) void tick();
      }, Math.max(0, Number(delayMs) || 0));
    }

    function markSessionsPollSuccess() {
      sessionsPollErrorStreak = 0;
    }

    function markSessionsPollFailure() {
      sessionsPollErrorStreak = Math.min(sessionsPollErrorStreak + 1, 20);
    }

    function markSecondaryPollSuccess() {
      secondaryPollErrorStreak = 0;
    }

    function markSecondaryPollFailure() {
      secondaryPollErrorStreak = Math.min(secondaryPollErrorStreak + 1, 20);
    }

    function resetStreaks() {
      sessionsPollErrorStreak = 0;
      secondaryPollErrorStreak = 0;
    }

    function disable() {
      sessionsPollingEnabled = false;
      secondaryPollingEnabled = false;
      cancelAll();
    }

    return Object.freeze({
      scheduleSessions,
      scheduleSecondary,
      disable,
      sessionsPollErrorStreak: () => sessionsPollErrorStreak,
      secondaryPollErrorStreak: () => secondaryPollErrorStreak,
      markSessionsPollSuccess,
      markSessionsPollFailure,
      markSecondaryPollSuccess,
      markSecondaryPollFailure,
      resetStreaks,
    });
  }

  function sessionsPollDelayMs(visibilityState) {
    return visibilityPollDelayMs(visibilityState, POLLING_INTERVALS.SESSION_POLL_VISIBLE_MS, POLLING_INTERVALS.SESSION_POLL_HIDDEN_MS);
  }

  function secondaryPollDelayMs(visibilityState) {
    return visibilityPollDelayMs(visibilityState, POLLING_INTERVALS.SECONDARY_POLL_VISIBLE_MS, POLLING_INTERVALS.SECONDARY_POLL_HIDDEN_MS);
  }

  function browserOffline(navigatorLike) {
    return typeof navigatorLike !== "undefined" && navigatorLike && navigatorLike.onLine === false;
  }

  function messagePollErrorDelayMs(errorStreak) {
    const streak = Number(errorStreak) || 0;
    if (!streak) return 0;
    const exponent = Math.min(6, Math.max(0, streak - 1));
    return Math.min(POLLING_INTERVALS.MESSAGE_POLL_ERROR_MAX_MS, POLLING_INTERVALS.MESSAGE_POLL_ERROR_MIN_MS * 2 ** exponent);
  }

  function networkRetryDelayMs({ normalDelayMs = 0, offline = false, errorStreak = 0 } = {}) {
    const normalDelay = Math.max(0, Number(normalDelayMs) || 0);
    const errorDelay = messagePollErrorDelayMs(errorStreak);
    return Math.max(normalDelay, offline ? POLLING_INTERVALS.MESSAGE_POLL_OFFLINE_MS : 0, errorDelay);
  }

  function messagePollDelayMs({ now = Date.now(), visibilityState = "visible", offline = false, errorStreak = 0, pollFastUntilMs = 0, turnOpen = false } = {}) {
    const errorDelay = messagePollErrorDelayMs(errorStreak);
    if (offline) return Math.max(POLLING_INTERVALS.MESSAGE_POLL_OFFLINE_MS, errorDelay);
    if (visibilityState === "hidden") return Math.max(POLLING_INTERVALS.MESSAGE_POLL_HIDDEN_MS, errorDelay);
    let delay = POLLING_INTERVALS.MESSAGE_POLL_IDLE_MS;
    if (now < pollFastUntilMs) delay = POLLING_INTERVALS.MESSAGE_POLL_FAST_MS;
    else if (turnOpen) delay = POLLING_INTERVALS.MESSAGE_POLL_RUNNING_MS;
    return Math.max(delay, errorDelay);
  }

  function normalizeMessagePollKickDelay({ requested = 0, now = Date.now(), visibilityState = "visible", offline = false, errorStreak = 0, pollFastUntilMs = 0, turnOpen = false } = {}) {
    const safeRequested = Math.max(0, Number(requested) || 0);
    const errorDelay = messagePollErrorDelayMs(errorStreak);
    if (offline || visibilityState === "hidden") {
      return Math.max(safeRequested, messagePollDelayMs({ now, visibilityState, offline, errorStreak, pollFastUntilMs, turnOpen }));
    }
    return Math.max(safeRequested, errorDelay);
  }

export { createAsyncEpoch, createPollingRuntime, POLLING_INTERVALS, sessionsPollDelayMs, secondaryPollDelayMs, browserOffline, messagePollErrorDelayMs, networkRetryDelayMs, messagePollDelayMs, normalizeMessagePollKickDelay };
