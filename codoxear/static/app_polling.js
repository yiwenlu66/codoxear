(function () {
  "use strict";

  const POLLING_INTERVALS = Object.freeze({
    SESSION_POLL_VISIBLE_MS: 3500,
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

  window.CodoxearPolling = Object.freeze({
    POLLING_INTERVALS,
    sessionsPollDelayMs,
    secondaryPollDelayMs,
    browserOffline,
    messagePollErrorDelayMs,
    messagePollDelayMs,
    normalizeMessagePollKickDelay,
  });
})();
