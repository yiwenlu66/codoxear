(function () {
  "use strict";

  function computeAppBaseUrl(locationLike) {
    const here = new URL(locationLike.href);
    const p0 = String(here.pathname || "/");
    if (p0.endsWith("/static/index.html")) {
      return new URL(p0.slice(0, -"/static/index.html".length) + "/", here.origin);
    }
    if (p0.endsWith("/static/")) {
      return new URL(p0.slice(0, -"/static/".length) + "/", here.origin);
    }
    return new URL(".", here);
  }

  const appBaseUrl = computeAppBaseUrl(window.location);

  function resolveAppUrl(path) {
    const s = String(path ?? "");
    const rel = s.startsWith("/") ? s.slice(1) : s;
    return new URL(rel, appBaseUrl).toString();
  }

  function sessionIdFromHash(locationLike = window.location) {
    const raw = String(locationLike.hash || "").startsWith("#") ? String(locationLike.hash || "").slice(1) : String(locationLike.hash || "");
    const params = new URLSearchParams(raw);
    const sid = params.get("session");
    return sid && sid.trim() ? sid.trim() : "";
  }

  function setSessionHash(sessionId, { locationLike = window.location, historyLike = window.history } = {}) {
    const raw = String(locationLike.hash || "").startsWith("#") ? String(locationLike.hash || "").slice(1) : String(locationLike.hash || "");
    const params = new URLSearchParams(raw);
    if (sessionId) params.set("session", sessionId);
    else params.delete("session");
    const next = params.toString();
    const target = `${locationLike.pathname}${locationLike.search}${next ? `#${next}` : ""}`;
    historyLike.replaceState(null, "", target);
    return target;
  }

  window.CodoxearUrls = Object.freeze({
    appBaseHref: appBaseUrl.toString(),
    resolveAppUrl,
    sessionIdFromHash,
    setSessionHash,
  });
})();
