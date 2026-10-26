  function requireNode(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`network status requires ${name}`);
    return value;
  }

  function browserOffline(navigatorLike) {
    return Boolean(navigatorLike && navigatorLike.onLine === false);
  }

  // One user-visible state ("not live") carried by an icon. The three reasons
  // below are evidence channels, not distinct UI states: the browser link
  // signal, a cache-served response that looks like a 200, and repeated
  // transport failures. They differ in how they are detected and cleared, so
  // they stay separate internally; only the hover/assistive text names them.
  const MESSAGES = Object.freeze({
    offline: "Offline — waiting for a network connection. Updates retry automatically.",
    cached: "Connection unavailable — showing cached data. Updates retry automatically.",
    failing: "Connection unavailable — retrying automatically.",
  });

  // Failures arrive per failed request from the transport layer (app_api), so
  // two concurrent failures inside one poll cycle trip the threshold. One flaky
  // request still shows nothing, and a caller's abort is not counted at all.
  function createNetworkStatusController({ status, statusText, navigatorLike = typeof navigator === "undefined" ? undefined : navigator } = {}) {
    requireNode(status, "status");
    requireNode(statusText, "statusText");
    let transportFailed = false;
    let staleServed = false;
    let consecutiveFailures = 0;
    const FAILURE_THRESHOLD = 2;

    function reason() {
      if (browserOffline(navigatorLike)) return "offline";
      if (staleServed) return "cached";
      if (transportFailed) return "failing";
      return "";
    }

    // The icon is static markup owned by the shell; this controller only
    // toggles visibility and writes the spoken/hover text. Unhiding before the
    // text changes keeps the polite live region announcing the new reason.
    function render() {
      const state = reason();
      const text = state ? MESSAGES[state] : "";
      if (!text) {
        if (statusText.textContent) statusText.textContent = "";
        if (status.hasAttribute("title")) status.removeAttribute("title");
        status.hidden = true;
        return state;
      }
      if (status.getAttribute("title") !== text) status.setAttribute("title", text);
      status.hidden = false;
      if (statusText.textContent !== text) statusText.textContent = text;
      return state;
    }

    function sync() {
      return render();
    }

    function reportFailure() {
      consecutiveFailures += 1;
      transportFailed = consecutiveFailures >= FAILURE_THRESHOLD;
      return render();
    }

    // The service worker answered from its offline cache: the payload was
    // usable, but no server responded. One stale answer is proof of
    // unreachability — no threshold needed. Without this, a cache-served 200
    // would read as health and the icon would clear on stale content. This is
    // the dominant evidence channel whenever the page is served offline.
    function reportStale() {
      staleServed = true;
      return render();
    }

    function reportSuccess() {
      transportFailed = false;
      staleServed = false;
      consecutiveFailures = 0;
      return render();
    }

    sync();
    return Object.freeze({ browserOffline, sync, reportFailure, reportStale, reportSuccess });
  }

export { browserOffline, createNetworkStatusController };
