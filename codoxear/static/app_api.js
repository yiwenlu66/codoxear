(function () {
  "use strict";

  const codoxearUrls = window.CodoxearUrls;
  if (!codoxearUrls || typeof codoxearUrls.resolveAppUrl !== "function") throw new Error("Codoxear URL helpers failed to load");
  const codoxearPerfHelpers = window.CodoxearPerf;
  if (!codoxearPerfHelpers || typeof codoxearPerfHelpers.pushSample !== "function") throw new Error("Codoxear performance helpers failed to load");

  const apiEtags = new Map();
  const API_NOT_MODIFIED = Symbol("api.notModified");

  function apiResponseNotModified(obj) {
    return !!(obj && obj[API_NOT_MODIFIED]);
  }

  function clearApiCache() {
    apiEtags.clear();
  }

  async function api(path, { method = "GET", body, signal } = {}) {
    const t0 = performance.now();
    const rawPath = String(path ?? "");
    const cacheableSessionsRequest = method === "GET" && rawPath === "/api/sessions";
    const opts = { method, headers: {}, signal };
    if (cacheableSessionsRequest && apiEtags.has(rawPath)) {
      opts.headers["If-None-Match"] = apiEtags.get(rawPath).etag;
    }
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const url = codoxearUrls.resolveAppUrl(path);
    const res = await fetch(url, opts);
    const dt = performance.now() - t0;
    if (rawPath === "/api/sessions" && method === "GET") codoxearPerfHelpers.pushSample("api_sessions_ms", dt);
    else if (rawPath.includes("/messages") && method === "GET") {
      if (rawPath.includes("init=1")) codoxearPerfHelpers.pushSample("api_messages_init_ms", dt);
      else codoxearPerfHelpers.pushSample("api_messages_poll_ms", dt);
    }
    if (res.status === 304 && cacheableSessionsRequest && apiEtags.has(rawPath)) {
      const cached = JSON.parse(apiEtags.get(rawPath).text);
      Object.defineProperty(cached, API_NOT_MODIFIED, { value: true });
      return cached;
    }
    const txt = await res.text();
    let obj;
    try {
      obj = JSON.parse(txt);
    } catch (e) {
      console.error("api: invalid json response", { path, url, method, txt });
      throw e;
    }
    if (!res.ok) throw Object.assign(new Error(obj.error || "request failed"), { status: res.status, obj });
    const etag = cacheableSessionsRequest ? res.headers.get("ETag") : null;
    if (etag) apiEtags.set(rawPath, { etag, text: txt });
    return obj;
  }

  window.CodoxearApi = Object.freeze({
    api,
    apiResponseNotModified,
    clearApiCache,
  });
})();
