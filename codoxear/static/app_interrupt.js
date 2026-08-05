(function (global) {
  "use strict";

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`interrupt controller dependency missing: ${name}`);
    return value;
  }

  function createInterruptController(options = {}) {
    const selectedSessionId = requireFunction(options.selectedSessionId, "selectedSessionId");
    const setToast = requireFunction(options.setToast, "setToast");
    const api = requireFunction(options.api, "api");
    const now = requireFunction(options.now, "now");
    const setPollFastUntilMs = requireFunction(options.setPollFastUntilMs, "setPollFastUntilMs");
    const kickPoll = requireFunction(options.kickPoll, "kickPoll");

    async function interruptSelectedSession() {
      const sessionId = selectedSessionId();
      if (!sessionId) return;
      try {
        setToast("interrupting...");
        await api(`/api/sessions/${sessionId}/interrupt`, { method: "POST" });
        setPollFastUntilMs(now() + 2500);
        kickPoll(0);
      } catch (error) {
        setToast(`interrupt error: ${error.message}`);
      }
    }

    return Object.freeze({ interruptSelectedSession });
  }

  global.CodoxearInterrupt = Object.freeze({ createInterruptController });
})(window);
