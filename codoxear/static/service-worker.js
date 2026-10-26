const HTML_CACHE = "codoxear-html-v1";
const STATIC_CACHE = "codoxear-static-v1";
const API_CACHE = "codoxear-api-v1";
const SW_META_CACHE = "codoxear-sw-meta-v1";
const APP_CACHES = new Set([HTML_CACHE, STATIC_CACHE, API_CACHE, SW_META_CACHE]);
const NETWORK_FIRST_MIN_TIMEOUT_MS = 3000;
const NETWORK_FIRST_MAX_TIMEOUT_MS = 10000;
// A delivery race is a latency policy, never evidence that the link is down.
const RETAINED_READ_TIMEOUT_MS = 60000;
const MAX_RETAINED_READS = 8;
const MAX_BACKGROUND_READS = 3;
const CACHE_WORK_TIMEOUT_MS = 5000;
const MAX_CACHE_READS = 8;
const MAX_CACHE_WRITES = 4;
const MAX_BACKOFF_KEYS = 128;
const STALE_HEADER = "X-Codoxear-Stale";
const pendingReads = new Map();
const refreshBackoff = new Map();
const cacheWrites = new Map();
let cacheReads = 0;
let backgroundReads = 0;
let activeReads = 0;
let fetchLatencyEwmaMs = 0;

function timeoutError() {
  return Object.assign(new Error("Retained read deadline exceeded"), { name: "TimeoutError" });
}

// Cache Storage offers no cancellation. Keep the slot until the actual work
// settles, even after its caller stops waiting, so a broken cache cannot spawn
// an unbounded chain of retained operations.
function withinBudget(work, ms) {
  let timer;
  return Promise.race([
    work,
    new Promise((_resolve, reject) => { timer = setTimeout(() => reject(timeoutError()), ms); }),
  ]).finally(() => clearTimeout(timer));
}

async function matchCached(cacheName, request, options) {
  if (cacheReads >= MAX_CACHE_READS) return undefined;
  cacheReads++;
  const work = Promise.resolve().then(async () => (await caches.open(cacheName)).match(request, options))
    .finally(() => { cacheReads--; });
  return withinBudget(work, CACHE_WORK_TIMEOUT_MS).catch(() => undefined);
}

function storeResponse(cacheName, request, response) {
  const key = `${cacheName}:${request.url}`;
  if (response.status !== 200 || cacheWrites.has(key) || cacheWrites.size >= MAX_CACHE_WRITES) return Promise.resolve();
  let copy;
  try { copy = response.clone(); } catch (_) { return Promise.resolve(); }
  const work = Promise.resolve().then(async () => (await caches.open(cacheName)).put(request, copy))
    .finally(() => cacheWrites.delete(key));
  cacheWrites.set(key, work);
  return withinBudget(work, CACHE_WORK_TIMEOUT_MS).catch(() => {});
}

function readKey(cacheName, request) {
  return JSON.stringify([cacheName, request.url, request.credentials, request.mode, request.cache,
    [...request.headers].sort(([a], [b]) => a.localeCompare(b))]);
}

function deferRefresh(key, reason) {
  const previous = refreshBackoff.get(key);
  const attempts = Math.min(5, (previous?.attempts || 0) + 1);
  refreshBackoff.delete(key);
  refreshBackoff.set(key, { attempts, reason, until: Date.now() + Math.min(30000, 1000 * 2 ** attempts) });
  if (refreshBackoff.size > MAX_BACKOFF_KEYS) refreshBackoff.delete(refreshBackoff.keys().next().value);
}

function transportResponse(kind) {
  return new Response(JSON.stringify({ error: kind === "deadline" ? "Read deadline exceeded" : "Retained read capacity exhausted" }), {
    status: kind === "deadline" ? 504 : 503,
    headers: { "Content-Type": "application/json", "X-Codoxear-Transport": kind },
  });
}

function recordFetchLatency(durationMs) {
  if (!(durationMs >= 0)) return;
  fetchLatencyEwmaMs = fetchLatencyEwmaMs ? fetchLatencyEwmaMs * 0.7 + durationMs * 0.3 : durationMs;
}

function networkFirstTimeoutMs() {
  if (!fetchLatencyEwmaMs) return NETWORK_FIRST_MIN_TIMEOUT_MS;
  return Math.min(NETWORK_FIRST_MAX_TIMEOUT_MS, Math.max(NETWORK_FIRST_MIN_TIMEOUT_MS, fetchLatencyEwmaMs * 2));
}

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  const claim = clients.claim();
  const prune = caches.keys().then((names) => Promise.all(
    names.filter((name) => name.startsWith("codoxear-") && !APP_CACHES.has(name)).map((name) => caches.delete(name))
  ));
  event.waitUntil(Promise.all([claim, prune, warmActivationCaches().catch(warnWarmupFailure)]));
});

function warnWarmupFailure(error) {
  console.warn("Codoxear offline cache warm-up failed", error);
}

async function warmUrl(url, cacheName) {
  if (cacheName === API_CACHE) {
    const request = new Request(url, { credentials: "same-origin" });
    const read = startRead(readKey(cacheName, request), request, cacheName, true);
    if (!read) return transportResponse("capacity");
    const response = await read.promise;
    await read.done;
    return response.clone();
  }
  const cache = await caches.open(cacheName);
  return fetchAndCache(new Request(url, { credentials: "same-origin" }), cache, {
    pruneVersions: cacheName === STATIC_CACHE,
  });
}

async function warmShell() {
  const shellUrl = new URL("./", self.registration.scope);
  const response = await warmUrl(shellUrl.href, HTML_CACHE);
  if (response.status !== 200) return;
  const html = (await response.text()).replace(/&amp;/g, "&");
  const assets = new Set();
  const attributes = /\b(?:src|href)\s*=\s*(["'])([^"'<>]*[?&]v=[^"'<>]*)\1/gi;
  for (const match of html.matchAll(attributes)) {
    const url = new URL(match[2], self.registration.scope);
    if (url.origin === self.location.origin && url.searchParams.has("v")) assets.add(url.href);
  }
  // The theme link is built by document.write, not a literal HTML attribute.
  // All shell assets share the server's memoized asset version.
  const firstAsset = assets.values().next().value;
  if (firstAsset) {
    const version = new URL(firstAsset).searchParams.get("v");
    for (const family of ["paper", "clay", "slate"]) {
      const url = new URL(`themes/${family}.css`, self.registration.scope);
      url.searchParams.set("v", version);
      assets.add(url.href);
    }
  }
  await Promise.all([...assets].map((url) => warmUrl(url, STATIC_CACHE).catch(warnWarmupFailure)));
}

async function warmActivationCaches() {
  await Promise.all([
    warmShell().catch(warnWarmupFailure),
    ...["api/me", "api/sessions"].map((path) =>
      warmUrl(new URL(path, self.registration.scope).href, API_CACHE).catch(warnWarmupFailure)
    ),
  ]);
}

async function fetchAndCache(request, cache, { pruneVersions = false } = {}) {
  const response = await fetch(request);
  if (response.status === 200) {
    await cache.put(request, response.clone());
    if (pruneVersions) {
      const url = new URL(request.url);
      const keys = await cache.keys();
      await Promise.all(keys.filter((key) => {
        const oldUrl = new URL(key.url);
        return oldUrl.pathname === url.pathname && oldUrl.searchParams.has("v") && oldUrl.searchParams.get("v") !== url.searchParams.get("v");
      }).map((key) => cache.delete(key)));
    }
  }
  return response;
}

function markStale(response, reason) {
  const headers = new Headers(response.headers);
  headers.set(STALE_HEADER, reason);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function startRead(key, request, cacheName, background) {
  if (pendingReads.has(key)) return pendingReads.get(key);
  if (activeReads >= MAX_RETAINED_READS || (background && backgroundReads >= MAX_BACKGROUND_READS)) return null;
  const controller = new AbortController();
  const startedAt = Date.now();
  const read = { storage: Promise.resolve() };
  activeReads++;
  if (background) backgroundReads++;
  let timer, expired = false;
  const deadline = new Promise((_resolve, reject) => {
    timer = setTimeout(() => {
      expired = true;
      reject(timeoutError());
      controller.abort();
    }, RETAINED_READ_TIMEOUT_MS);
  });
  // The shared request has its own lifetime: cancelling one page consumer must
  // not abort another consumer or a cache refresh already retained by the worker.
  const work = Promise.resolve().then(async () => {
    const response = await fetch(request, { signal: controller.signal });
    // Finite API JSON completes at usable bytes. Native streams and derivatives
    // are excluded by routing; non-JSON document/media bodies stay native.
    if (cacheName === API_CACHE && /(?:application|text)\/(?:[^;]+\+)?json\b/i.test(response.headers.get("Content-Type") || "")) {
      const bytes = await response.arrayBuffer();
      return new Response([204, 205, 304].includes(response.status) ? null : bytes, { status: response.status, statusText: response.statusText, headers: response.headers });
    }
    return response;
  }).finally(() => {
    activeReads--;
    if (background) backgroundReads--;
  });
  read.promise = Promise.race([work, deadline]).then((response) => {
    const duration = Date.now() - startedAt;
    recordFetchLatency(duration);
    if (duration >= NETWORK_FIRST_MIN_TIMEOUT_MS) deferRefresh(key, "timeout");
    else refreshBackoff.delete(key);
    read.storage = storeResponse(cacheName, request, response);
    return response;
  }, (error) => {
    // Locally chosen deadlines are slowness evidence only. Native fetch/body
    // rejection is a failed transfer, not a statement about cache freshness.
    const cancelled = error?.name === "AbortError";
    deferRefresh(key, expired || cancelled ? "timeout" : "suspect");
    if (expired) return transportResponse("deadline");
    throw error;
  }).finally(() => {
    clearTimeout(timer);
  });
  read.done = read.promise.then(() => read.storage).catch(() => {}).finally(() => pendingReads.delete(key));
  pendingReads.set(key, read);
  return read;
}

function networkFirst(event, cacheName, matchOptions, onFallback) {
  let retain;
  event.waitUntil(new Promise((resolve) => { retain = resolve; }));
  return networkFirstResponse(event.request, cacheName, matchOptions, onFallback, retain)
    .catch((error) => { retain(); throw error; });
}

async function networkFirstResponse(request, cacheName, matchOptions, onFallback, retain) {
  const key = readKey(cacheName, request);
  // Read cache concurrently with starting a foreground request. A stalled cache
  // lookup must not delay a valid live answer or consume the network budget.
  const cachedPromise = matchCached(cacheName, request, matchOptions);
  const offline = self.navigator?.onLine === false;
  const cooldown = refreshBackoff.get(key);
  const deferred = cooldown && Date.now() < cooldown.until;
  let cached;
  if (offline || cooldown) {
    cached = await cachedPromise;
    if (cached) {
      const read = pendingReads.get(key) || (!offline && !deferred ? startRead(key, request, cacheName, true) : null);
      retain(read?.done);
      onFallback?.();
      return markStale(cached, offline ? "offline" : cooldown.reason);
    }
  }
  const read = startRead(key, request, cacheName, false);
  if (!read) {
    retain();
    cached = cached || await cachedPromise;
    if (cached) { onFallback?.(); return markStale(cached, "timeout"); }
    return transportResponse("capacity");
  }
  retain(read.done);
  let timer, settled = false;
  try {
    const delivered = read.promise.then((response) => response.clone());
    // A cached answer may win after the delivery budget. With no cached answer
    // there is no useful short timeout: await the complete retained lifecycle.
    const fallback = new Promise((resolve) => {
      timer = setTimeout(async () => {
        const hit = cached || await cachedPromise;
        if (hit && !settled) {
          deferRefresh(key, "timeout");
          onFallback?.();
          resolve(markStale(hit, "timeout"));
        }
      }, networkFirstTimeoutMs());
    });
    const response = await Promise.race([delivered, fallback]);
    if (response.headers.get("X-Codoxear-Transport") === "deadline") {
      cached = cached || await cachedPromise;
      if (cached) { onFallback?.(); return markStale(cached, "timeout"); }
    }
    return response;
  } catch (error) {
    cached = cached || await cachedPromise;
    if (cached) { onFallback?.(); return markStale(cached, error?.name === "AbortError" ? "timeout" : "suspect"); }
    throw error;
  } finally {
    settled = true;
    clearTimeout(timer);
  }
}

async function versionedStatic(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);
  return cached || fetchAndCache(request, cache, { pruneVersions: true });
}

function recordNavigation(request, offline) {
  const marker = JSON.stringify({ ts: Date.now(), offline, url: request.url });
  const markerNames = ["last-navigation"];
  if (offline) markerNames.push("last-offline-navigation");
  return caches.open(SW_META_CACHE).then((cache) => Promise.all(markerNames.map((name) => cache.put(
    new Request(new URL(`sw-meta/${name}`, self.registration.scope)),
    new Response(marker, { headers: { "content-type": "application/json" } })
  ))));
}

function startNavigationMarker(event, request) {
  let finish;
  event.waitUntil(new Promise((resolve) => {
    finish = (offline) => {
      try {
        resolve(recordNavigation(request, offline).catch(() => {}));
      } catch (_error) {
        resolve();
      }
    };
  }));
  return finish;
}

async function navigationNetworkFirst(event, finishMarker) {
  let offline = false;
  try {
    return await networkFirst(event, HTML_CACHE, { ignoreSearch: true }, () => { offline = true; });
  } finally {
    // Diagnostics must never alter navigation delivery, including a failed fallback.
    finishMarker(offline);
  }
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  const scopePath = new URL(self.registration.scope).pathname;
  if (request.method !== "GET" || url.origin !== self.location.origin || !url.pathname.startsWith(scopePath)) return;
  const pathname = `/${url.pathname.slice(scopePath.length)}`;
  // Mutation verification must never read or populate either HTTP or SW caches.
  if (request.cache === "no-store" || /^\/api\/sessions\/[^/]+\/attachments\/upload-status$/.test(pathname)) return;
  // Native media/derivative clients own streaming and Range semantics. These
  // conversions can spend unbounded time before headers, so a reachability
  // race or cached full response is the wrong transport policy for them.
  if (/^\/api\/(?:sessions\/[^/]+\/file|files)\/(?:office|video)_preview$/.test(pathname)) return;
  // Cache.match has no partial-response semantics: never answer a Range
  // request with a cached full 200, and never bring 206 into a cache write.
  if (request.headers && typeof request.headers.has === "function" && request.headers.has("range")) return;
  // This is the SSE stream, not the JSON /messages/live polling route.
  if (/^\/api\/sessions\/[^/]+\/live$/.test(pathname)) return;
  if (request.mode === "navigate") {
    const finishMarker = startNavigationMarker(event, request);
    event.respondWith(navigationNetworkFirst(event, finishMarker));
  } else if (pathname.startsWith("/api/")) {
    event.respondWith(networkFirst(event, API_CACHE));
  } else if (url.searchParams.has("v")) {
    event.respondWith(versionedStatic(request));
  } else {
    const cachePromise = caches.open(STATIC_CACHE);
    const revalidation = cachePromise.then((cache) => fetchAndCache(request, cache));
    // Offline revalidation may fail while the cached response remains usable.
    event.waitUntil(revalidation.catch(() => {}));
    event.respondWith(cachePromise.then(async (cache) => (await cache.match(request)) || revalidation));
  }
});

self.addEventListener("push", (event) => {
  let payload = {};
  try {
    payload = event.data ? event.data.json() : {};
  } catch (_error) {
    payload = {};
  }
  const sessionName = typeof payload.session_display_name === "string" && payload.session_display_name ? payload.session_display_name : "Codoxear";
  const body = typeof payload.notification_text === "string" && payload.notification_text ? payload.notification_text : "New final response";
  const sessionId = typeof payload.session_id === "string" ? payload.session_id : "";
  const targetUrl = new URL("./", self.registration.scope);
  if (sessionId) {
    targetUrl.hash = `session=${encodeURIComponent(sessionId)}`;
  }
  event.waitUntil(
    self.registration.showNotification(sessionName, {
      body,
      tag: sessionId || "codoxear-final-response",
      data: { url: targetUrl.toString() },
      renotify: true,
    })
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const rawUrl = event.notification && event.notification.data ? event.notification.data.url : "";
  const target = typeof rawUrl === "string" && rawUrl ? rawUrl : new URL("./", self.registration.scope).toString();
  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true }).then(async (windows) => {
      for (const client of windows) {
        if ("focus" in client) {
          if ("navigate" in client) {
            const navigated = await client.navigate(target);
            return (navigated || client).focus();
          }
          return client.focus();
        }
      }
      return clients.openWindow(target);
    })
  );
});
