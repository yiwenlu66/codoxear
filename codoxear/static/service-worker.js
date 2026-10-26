const HTML_CACHE = "codoxear-html-v1";
const STATIC_CACHE = "codoxear-static-v1";
const API_CACHE = "codoxear-api-v1";
const SW_META_CACHE = "codoxear-sw-meta-v1";
const APP_CACHES = new Set([HTML_CACHE, STATIC_CACHE, API_CACHE, SW_META_CACHE]);
const NETWORK_FIRST_MIN_TIMEOUT_MS = 3000;
const NETWORK_FIRST_MAX_TIMEOUT_MS = 10000;
const NETWORK_SUSPECT_WINDOW_MS = 30000;
// A lost race starts as unproven slowness: the retained fetch may still
// succeed. Only when no network-first fetch has settled successfully for this
// long does a hang escalate to "unreachable" and arm the suspect latch.
const HANG_SUSPECT_AFTER_MS = 8000;
// Marks a cache-served fallback so the page can tell "server answered" from
// "service worker served stale data": behind a cache-serving worker, a
// resolved 200 is evidence about the cache, not the network. The value names
// why the cache answered:
//   offline — the browser reports no link (navigator.onLine === false)
//   suspect — a fetch rejected outright, or a hang outlasted slowness
//   timeout — the delivery-bounding race lost, but the fetch is still running;
//             this is evidence of slowness, never of unreachability
const STALE_HEADER = "X-Codoxear-Stale";
let networkSuspectUntil = 0;
// Clock time of the first race timeout since the last successful network-first
// fetch; 0 while every recent fetch has settled. Any success or rejection
// resets it — success proves reachability, rejection arms the latch directly.
let unsettledNetworkSince = 0;
// EWMA of successful network-first fetch durations; 0 until the first sample.
// The race timeout adapts to the observed connection so a consistently slow
// link stops losing every race, while a sample-less worker keeps the 3s bound
// that the offline cold-boot budget relies on.
let fetchLatencyEwmaMs = 0;

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

async function networkFirst(event, cacheName, matchOptions, onFallback) {
  const request = event.request;
  const cache = await caches.open(cacheName);
  if (self.navigator && self.navigator.onLine === false) {
    const cached = await cache.match(request, matchOptions);
    if (cached) {
      onFallback?.();
      return markStale(cached, "offline");
    }
  }
  if (Date.now() < networkSuspectUntil) {
    const cached = await cache.match(request, matchOptions);
    if (cached) {
      onFallback?.();
      const startedAt = Date.now();
      event.waitUntil(fetchAndCache(request, cache).then(() => {
        recordFetchLatency(Date.now() - startedAt);
        networkSuspectUntil = 0;
        unsettledNetworkSince = 0;
      }).catch(() => {}));
      return markStale(cached, "suspect");
    }
  }
  // Airplane + VPN tails and captive portals can hang instead of rejecting.
  // Bound response delivery, but retain the fetch so its outcome decides the
  // verdict: rejection proves unreachability and arms the suspect latch, while
  // a late success proves mere slowness, refreshes the cache, and feeds the
  // adaptive race bound. Losing the race itself arms nothing.
  const startedAt = Date.now();
  let rejected = false;
  const network = fetchAndCache(request, cache).then((response) => {
    recordFetchLatency(Date.now() - startedAt);
    networkSuspectUntil = 0;
    unsettledNetworkSince = 0;
    return response;
  }, (error) => {
    rejected = true;
    networkSuspectUntil = Date.now() + NETWORK_SUSPECT_WINDOW_MS;
    unsettledNetworkSince = 0;
    throw error;
  });
  event.waitUntil(network.catch(() => {}));
  let timer;
  try {
    return await Promise.race([
      network,
      new Promise((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error("Network-first request timed out")), networkFirstTimeoutMs());
      }),
    ]);
  } catch (error) {
    onFallback?.();
    const cached = await cache.match(request, matchOptions);
    if (cached) {
      if (rejected) return markStale(cached, "suspect");
      const now = Date.now();
      if (!unsettledNetworkSince) unsettledNetworkSince = now;
      // A fetch that never settles outlasts slowness: escalate the hang to
      // unreachability so cache fallbacks stop masquerading as a live server.
      if (now - unsettledNetworkSince >= HANG_SUSPECT_AFTER_MS) {
        networkSuspectUntil = now + NETWORK_SUSPECT_WINDOW_MS;
        return markStale(cached, "suspect");
      }
      return markStale(cached, "timeout");
    }
    throw error;
  } finally {
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
