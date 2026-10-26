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
  const API_RESPONSE_STALE = Symbol("api.responseStale");

  const API_TRANSPORT_LIVE = "live";
  const API_TRANSPORT_STALE = "stale";
  const API_TRANSPORT_FAILED = "failed";

  // Connection state belongs to the client, not to whichever session's request
  // happens to run: every response here is transport evidence, so a session
  // switch, a file read, or a send updates the indicator as promptly as a poll
  // tick does. This layer is the only producer of that evidence.
  let transportReporter = null;

  function setTransportReporter(reporter) {
    transportReporter = typeof reporter === "function" ? reporter : null;
  }

  function reportTransport(kind) {
    if (!transportReporter) return;
    try {
      transportReporter(kind);
    } catch (error) {
      // A broken reporter must never turn into a failed request.
      console.error("api: transport reporter failed", error);
    }
  }

  function apiResponseNotModified(obj) {
    return !!(obj && obj[API_NOT_MODIFIED]);
  }

  // True when the service worker answered from its offline cache instead of
  // the server (marked with X-Codoxear-Stale). The payload is cached data
  // whatever the mark's reason, so callers must not treat it as fresh.
  function apiResponseStale(obj) {
    return !!(obj && obj[API_RESPONSE_STALE]);
  }

  function clearApiCache() {
    apiEtags.clear();
  }

  const RESPONSE_BODY_TIMEOUT_MS = 60000;

  function requestTimeout() {
    return Object.assign(new Error("Request deadline exceeded"), { name: "TimeoutError", code: "REQUEST_TIMEOUT" });
  }

  async function api(path, { method = "GET", body, rawBody, headers, cache, signal, timeoutMs } = {}) {
    if (body !== undefined && rawBody !== undefined) throw new TypeError("body and rawBody are mutually exclusive");
    if (timeoutMs !== undefined && !(Number.isFinite(timeoutMs) && timeoutMs > 0)) {
      throw new TypeError("timeoutMs must be a positive finite duration");
    }
    const t0 = performance.now();
    const rawPath = String(path ?? "");
    const authoritative = /\/attachments\/upload-status(?:\?|$)/.test(rawPath);
    const cacheMode = authoritative ? "no-store" : cache;
    const cacheableSessionsRequest = method === "GET" && rawPath === "/api/sessions" && cacheMode !== "no-store";
    const requestHeaders = {};
    // Keep a plain header record for existing callers, accepting Headers and tuples too.
    if (headers) {
      const entries = typeof headers[Symbol.iterator] === "function" ? headers : Object.entries(headers);
      for (const [name, value] of entries) requestHeaders[name] = value;
    }
    const hasHeader = (name) => Object.keys(requestHeaders).some((key) => key.toLowerCase() === name.toLowerCase());
    const opts = { method, headers: requestHeaders, signal };
    if (cacheMode !== undefined) opts.cache = cacheMode;
    if (cacheableSessionsRequest && apiEtags.has(rawPath) && !hasHeader("If-None-Match")) {
      requestHeaders["If-None-Match"] = apiEtags.get(rawPath).etag;
    }
    if (body !== undefined) {
      if (!hasHeader("Content-Type")) requestHeaders["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    } else if (rawBody !== undefined) {
      if (!hasHeader("Content-Type")) requestHeaders["Content-Type"] = "application/octet-stream";
      opts.body = rawBody;
    }
    const url = CodoxearUrls.resolveAppUrl(path);
    // No default total deadline: uploading a file may legitimately take minutes.
    // The finite response-body budget starts only after response headers arrive.
    const controller = typeof AbortController === "undefined" ? null : new AbortController();
    if (controller) opts.signal = controller.signal;
    let timer, timedOut = false, rejectDeadline;
    const deadline = new Promise((_resolve, reject) => { rejectDeadline = reject; });
    const armDeadline = (ms) => {
      timer = setTimeout(() => {
        timedOut = true;
        rejectDeadline(requestTimeout());
        controller?.abort();
      }, ms);
    };
    const cancel = () => {
      controller?.abort();
      rejectDeadline(Object.assign(new Error("Request cancelled"), { name: "AbortError" }));
    };
    if (signal) {
      signal.addEventListener("abort", cancel, { once: true });
      if (signal.aborted) cancel();
    }
    if (timeoutMs !== undefined) armDeadline(timeoutMs);
    let res, stale, responseComplete = false;
    try {
      res = await Promise.race([fetch(url, opts), deadline]);
      stale = res.headers.get("X-Codoxear-Stale");
      const transport = res.headers.get("X-Codoxear-Transport");
      if (transport === "deadline" || transport === "capacity") {
        if (transport === "deadline") throw requestTimeout();
        throw Object.assign(new Error("Retained read capacity exhausted"), { name: "ReadCapacityError", code: "READ_CAPACITY" });
      }
      if (timeoutMs === undefined && res.body) armDeadline(RESPONSE_BODY_TIMEOUT_MS);
      const txt = await Promise.race([res.text(), deadline]);
      responseComplete = true;
      // Complete bytes prove contact, regardless of HTTP status or JSON validity.
      // Cached bytes only describe freshness; only failure-backed/offline cache
      // marks feed the connection indicator. A delivery timeout is no verdict.
      if (!stale) reportTransport(API_TRANSPORT_LIVE);
      else if (stale === "offline" || stale === "suspect") reportTransport(API_TRANSPORT_STALE);
      const dt = performance.now() - t0;
      if (rawPath === "/api/sessions" && method === "GET") codoxearPerfHelpers.pushSample("api_sessions_ms", dt);
      else if (rawPath.includes("/messages") && method === "GET") {
        codoxearPerfHelpers.pushSample(rawPath.includes("init=1") ? "api_messages_init_ms" : "api_messages_poll_ms", dt);
      }
      if (res.status === 304 && cacheableSessionsRequest && apiEtags.has(rawPath)) {
        const cached = JSON.parse(apiEtags.get(rawPath).text);
        Object.defineProperty(cached, API_NOT_MODIFIED, { value: true });
        if (stale) Object.defineProperty(cached, API_RESPONSE_STALE, { value: true });
        return cached;
      }
      let obj;
      try {
        obj = JSON.parse(txt);
      } catch (error) {
        console.error("api: invalid json response", { path, url, method });
        throw error;
      }
      if (!res.ok) throw Object.assign(new Error(obj.error || "request failed"), { status: res.status, obj });
      if (stale && obj && typeof obj === "object") Object.defineProperty(obj, API_RESPONSE_STALE, { value: true });
      const etag = cacheableSessionsRequest && !stale ? res.headers.get("ETag") : null;
      if (etag) apiEtags.set(rawPath, { etag, text: txt });
      return obj;
    } catch (error) {
      // Body stream failures count just like fetch failures; HTTP/JSON errors,
      // caller cancellation and our own delivery deadline do not imply outage.
      if (timedOut) throw requestTimeout();
      if (!responseComplete && !signal?.aborted && error?.name !== "AbortError" && error?.name !== "TimeoutError"
          && error?.name !== "ReadCapacityError" && !error?.status && error?.name !== "SyntaxError") {
        reportTransport(API_TRANSPORT_FAILED);
      }
      throw error;
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      signal?.removeEventListener("abort", cancel);
    }
  }

export { pushSample, summarize, api, apiResponseNotModified, apiResponseStale, clearApiCache, setTransportReporter };
