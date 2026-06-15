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

  window.CodoxearUrls = Object.freeze({
    appBaseHref: appBaseUrl.toString(),
    resolveAppUrl,
  });
})();
