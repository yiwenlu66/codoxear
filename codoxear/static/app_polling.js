  "use strict";

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

  function createSecondaryPollController(options = {}) {
    function requireSecondaryPollFunction(value, name) {
      if (typeof value !== "function") throw new TypeError(`secondary poll controller dependency missing: ${name}`);
      return value;
    }

    const isDisposed = requireSecondaryPollFunction(options.isDisposed, "isDisposed");
    const isPollingEnabled = requireSecondaryPollFunction(options.isPollingEnabled, "isPollingEnabled");
    const stopPolling = requireSecondaryPollFunction(options.stopPolling, "stopPolling");
    const setTimer = requireSecondaryPollFunction(options.setTimer, "setTimer");
    const scheduleTimer = requireSecondaryPollFunction(options.setTimeout, "setTimeout");
    const runTick = requireSecondaryPollFunction(options.runTick, "runTick");
    const delayForPoll = requireSecondaryPollFunction(options.delayForPoll, "delayForPoll");

    function scheduleSecondaryPoll(delayMs = delayForPoll()) {
      if (isDisposed() || !isPollingEnabled()) return;
      stopPolling();
      setTimer(
        scheduleTimer(() => {
          setTimer(null);
          void runTick();
        }, Math.max(0, Number(delayMs) || 0))
      );
    }

    return Object.freeze({ scheduleSecondaryPoll });
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

export { createSecondaryPollController, POLLING_INTERVALS, sessionsPollDelayMs, secondaryPollDelayMs, browserOffline, messagePollErrorDelayMs, networkRetryDelayMs, messagePollDelayMs, normalizeMessagePollKickDelay };
