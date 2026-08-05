(function (global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`secondary poll controller dependency missing: ${name}`);
    return value;
  }

  function createSecondaryPollController(options = {}) {
    const isDisposed = requireFunction(options.isDisposed, "isDisposed");
    const isPollingEnabled = requireFunction(options.isPollingEnabled, "isPollingEnabled");
    const stopPolling = requireFunction(options.stopPolling, "stopPolling");
    const setTimer = requireFunction(options.setTimer, "setTimer");
    const scheduleTimer = requireFunction(options.setTimeout, "setTimeout");
    const runTick = requireFunction(options.runTick, "runTick");
    const delayForPoll = requireFunction(options.delayForPoll, "delayForPoll");

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

  global.CodoxearSecondaryPoll = Object.freeze({ createSecondaryPollController });
})(window);
