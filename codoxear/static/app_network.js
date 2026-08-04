(function () {
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
      transportFailed = true;
      return render();
    }

    function reportSuccess() {
      transportFailed = false;
      return render();
    }

    sync();
    return Object.freeze({ browserOffline, sync, reportFailure, reportSuccess });
  }

  window.CodoxearNetwork = Object.freeze({ browserOffline, createNetworkStatusController });
})();
