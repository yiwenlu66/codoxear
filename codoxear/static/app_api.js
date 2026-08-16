import * as CodoxearUrls from "./app_application.js";
const perfWindow = 200;
  const perfSamples = new Map();

  function pushSample(name, valueMs) {
    if (!(valueMs >= 0)) return;
    const arr = perfSamples.get(name) || [];
    arr.push(valueMs);
    if (arr.length > perfWindow) arr.splice(0, arr.length - perfWindow);
    perfSamples.set(name, arr);
  }

  function percentile(sorted, p) {
    if (!sorted.length) return 0;
    if (sorted.length === 1) return sorted[0];
    const pos = Math.max(0, Math.min(1, p)) * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, sorted.length - 1);
    const frac = pos - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
  }

  function summarize() {
    const out = {};
    for (const [k, arr] of perfSamples.entries()) {
      if (!arr.length) continue;
      const s = arr.slice().sort((a, b) => a - b);
      out[k] = {
        count: s.length,
        p50_ms: Math.round(percentile(s, 0.5) * 100) / 100,
        p95_ms: Math.round(percentile(s, 0.95) * 100) / 100,
        max_ms: Math.round(s[s.length - 1] * 100) / 100,
        last_ms: Math.round(arr[arr.length - 1] * 100) / 100,
      };
    }
    return out;
  }

  const codoxearPerfHelpers = { pushSample, summarize };

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
    const url = CodoxearUrls.resolveAppUrl(path);
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

export { pushSample, summarize, api, apiResponseNotModified, clearApiCache };
