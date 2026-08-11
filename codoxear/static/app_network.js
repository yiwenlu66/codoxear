  "use strict";

  function requireNode(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`network status requires ${name}`);
    return value;
  }

  function browserOffline(navigatorLike) {
    return Boolean(navigatorLike && navigatorLike.onLine === false);
  }

  function createNetworkStatusController({ banner, navigatorLike = typeof navigator === "undefined" ? undefined : navigator } = {}) {
    requireNode(banner, "banner");
    let transportFailed = false;
    let consecutiveFailures = 0;
    const FAILURE_THRESHOLD = 2;

    function render() {
      const offline = browserOffline(navigatorLike);
      const text = offline
        ? "Offline — waiting for a network connection. Updates retry automatically."
        : transportFailed
          ? "Connection unavailable — retrying automatically."
          : "";
      banner.textContent = text;
      banner.hidden = !text;
      banner.setAttribute("aria-hidden", text ? "false" : "true");
      return text;
    }

    function sync() {
      return render();
    }

    function reportFailure() {
      consecutiveFailures += 1;
      transportFailed = consecutiveFailures >= FAILURE_THRESHOLD;
      return render();
    }

    function reportSuccess() {
      transportFailed = false;
      consecutiveFailures = 0;
      return render();
    }

    sync();
    return Object.freeze({ browserOffline, sync, reportFailure, reportSuccess });
  }

export { browserOffline, createNetworkStatusController };
