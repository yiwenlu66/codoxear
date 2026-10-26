"""Execute the real worker against an in-memory Cache Storage/browser sandbox."""
import json
import subprocess
from pathlib import Path


WORKER = Path(__file__).resolve().parents[1] / "codoxear/static/service-worker.js"
HARNESS = r"""
const vm = require('node:vm');
const assert = require('node:assert/strict');
const origin = 'https://codoxear.example';
const listeners = {}, stores = new Map(), fetches = [], notifications = [], clientCalls = [];
let network = async () => new Response('online');
let skips = 0, puts = 0;
const warnings = [];
const keyUrl = request => typeof request === 'string' ? new URL(request, origin).href : request.url;
const caches = {
  async open(name) {
    if (!stores.has(name)) stores.set(name, new Map());
    const store = stores.get(name);
    return {
      async match(request, options = {}) {
        const url = keyUrl(request);
        for (const [key, response] of store) {
          if (key === url || (options.ignoreSearch && key.split('?')[0] === url.split('?')[0])) return response.clone();
        }
      },
      async put(request, response) { puts++; store.set(keyUrl(request), response.clone()); },
      async keys() { return [...store.keys()].map(url => new Request(url)); },
      async delete(request) { return store.delete(keyUrl(request)); },
    };
  },
  async keys() { return [...stores.keys()]; },
  async delete(name) { return stores.delete(name); },
};
const clients = {
  async claim() { clientCalls.push('claim'); },
  async matchAll() { return []; },
  async openWindow(url) { clientCalls.push(url); },
};
const self = {
  navigator: { onLine: true },
  location: { origin },
  registration: { scope: origin + '/', async showNotification(title, options) { notifications.push({ title, options }); } },
  async skipWaiting() { skips++; },
  addEventListener(type, listener) { listeners[type] = listener; },
};
let clock = 0, timerId = 0;
const timers = new Map();
function setTimeout(fn, delay) { const id = ++timerId; timers.set(id, { fn, at: clock + delay }); return id; }
function clearTimeout(id) { timers.delete(id); }
async function flush() { for (let i = 0; i < 60; i++) await Promise.resolve(); await new Promise(setImmediate); }
async function advance(ms) {
  clock += ms;
  for (const [id, timer] of [...timers]) {
    if (timer.at <= clock) { timers.delete(id); timer.fn(); }
  }
  await flush();
}
class FakeDate extends Date { static now() { return clock; } }
const ctx = { self, clients, caches, URL, Request, Response, Headers, AbortController, Date: FakeDate, setTimeout, clearTimeout, console: { warn: (...args) => warnings.push(args) }, fetch: async (request, options) => { fetches.push(request.url); return network(request, options); } };
vm.createContext(ctx);
vm.runInContext(WORKER_SOURCE, ctx);
function dispatch(type, extra = {}) {
  const waits = [];
  let response;
  listeners[type]({ ...extra, waitUntil(promise) { waits.push(promise); }, respondWith(promise) { response = promise; } });
  return { response, waits, async finish() { await Promise.all(waits); } };
}
function request(path, options = {}) {
  return { url: new URL(path, origin).href, method: options.method || 'GET', mode: options.mode || 'cors', cache: options.cache || 'default', signal: options.signal,
    headers: options.headers instanceof Headers ? options.headers : new Headers(options.headers || {}) };
}
async function get(path, options = {}) {
  const event = dispatch('fetch', { request: request(path, options) });
  assert.ok(event.response, 'expected interception');
  const response = await event.response;
  await event.finish();
  return response;
}
async function cached(name, path) { return (await caches.open(name)).match(request(path)); }
function offline() { network = async () => { throw new TypeError('offline'); }; }
"""


def run_worker(body: str) -> None:
    program = "const WORKER_SOURCE = " + json.dumps(WORKER.read_text()) + ";\n" + HARNESS
    program += "\n(async () => {\n" + body + "\nconsole.log('WORKER_DONE');\n})().catch(e => { console.error(e); process.exit(1); });"
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "WORKER_DONE" in result.stdout, "scenario did not complete (unsettled promise)"


def test_navigation_network_first_and_ignore_search_offline_fallback() -> None:
    run_worker("""
      assert.equal(await (await get('/?entry=one', { mode: 'navigate' })).text(), 'online');
      assert.equal(await (await cached('codoxear-html-v1', '/?entry=one')).text(), 'online');
      network = async () => new Response('fresh shell');
      assert.equal(await (await get('/?entry=one', { mode: 'navigate' })).text(), 'fresh shell');
      offline();
      assert.equal(await (await get('/?entry=two', { mode: 'navigate' })).text(), 'fresh shell');
      await assert.rejects(get('/uncached', { mode: 'navigate' }), /offline/);
    """)


def test_navigation_marker_preserves_last_offline_fallback_across_network_recovery() -> None:
    run_worker("""
      network = async () => new Response('network shell');
      assert.equal(await (await get('/?first=1', { mode: 'navigate' })).text(), 'network shell');
      let marker = await (await cached('codoxear-sw-meta-v1', '/sw-meta/last-navigation')).json();
      assert.equal(marker.offline, false);
      assert.equal(marker.url, origin + '/?first=1');
      assert.equal(await cached('codoxear-sw-meta-v1', '/sw-meta/last-offline-navigation'), undefined);

      offline();
      assert.equal(await (await get('/?second=2', { mode: 'navigate' })).text(), 'network shell');
      marker = await (await cached('codoxear-sw-meta-v1', '/sw-meta/last-navigation')).json();
      const offlineMarker = await (await cached('codoxear-sw-meta-v1', '/sw-meta/last-offline-navigation')).json();
      assert.equal(marker.offline, true);
      assert.equal(marker.url, origin + '/?second=2');
      assert.deepEqual(offlineMarker, marker);
      assert.equal(typeof offlineMarker.ts, 'number');

      network = async () => new Response('recovered shell');
      // The first recovery request serves cache while its revalidation clears the latch.
      assert.equal(await (await get('/?third=3', { mode: 'navigate' })).text(), 'recovered shell');
      const recoveryFallback = await (await cached('codoxear-sw-meta-v1', '/sw-meta/last-offline-navigation')).json();
      assert.equal(recoveryFallback.offline, true);
      assert.equal(recoveryFallback.url, origin + '/?second=2');
      assert.equal(await (await get('/?fourth=4', { mode: 'navigate' })).text(), 'recovered shell');
      marker = await (await cached('codoxear-sw-meta-v1', '/sw-meta/last-navigation')).json();
      const retainedOfflineMarker = await (await cached('codoxear-sw-meta-v1', '/sw-meta/last-offline-navigation')).json();
      assert.equal(marker.offline, false);
      assert.equal(marker.url, origin + '/?fourth=4');
      assert.deepEqual(retainedOfflineMarker, recoveryFallback);

      const openCache = caches.open;
      caches.open = async (name) => {
        if (name === 'codoxear-sw-meta-v1') throw new Error('marker storage unavailable');
        return openCache(name);
      };
      const markerFailure = dispatch('fetch', { request: request('/marker-write-failure', { mode: 'navigate' }) });
      assert.equal(await (await markerFailure.response).text(), 'recovered shell');
      await markerFailure.finish();
    """)


def test_api_200_snapshots_survive_offline_with_exact_query_keys() -> None:
    run_worker("""
      for (const path of ['/api/me', '/api/sessions', '/api/sessions/s1/messages/tail?limit=60', '/api/sessions/s1/messages/history?before=c1']) {
        network = async () => new Response(JSON.stringify({ path }));
        await get(path);
        offline();
        assert.deepEqual(await (await get(path)).json(), { path });
      }
      await assert.rejects(get('/api/sessions/s1/messages/tail?limit=10'), /offline/);
      network = async () => new Response('new sessions');
      await advance(30000);
      await get('/api/sessions'); // Cached delivery plus successful background recovery.
      assert.equal(await (await get('/api/sessions')).text(), 'new sessions');
    """)


def test_cache_fallback_responses_are_marked_stale_with_a_reason() -> None:
    run_worker("""
      network = async () => new Response('payload', { headers: { 'ETag': 'v1' } });
      const live = await get('/api/sessions');
      assert.equal(live.headers.get('X-Codoxear-Stale'), null);
      assert.equal(await live.text(), 'payload');

      // Network rejection -> cached fallback: marked suspect, headers and body preserved.
      offline();
      const rejected = await get('/api/sessions');
      assert.equal(rejected.status, 200);
      assert.equal(rejected.headers.get('X-Codoxear-Stale'), 'suspect');
      assert.equal(rejected.headers.get('ETag'), 'v1');
      assert.equal(await rejected.text(), 'payload');

      // networkSuspect window: cache served immediately, still marked suspect.
      const suspect = await get('/api/sessions');
      assert.equal(suspect.headers.get('X-Codoxear-Stale'), 'suspect');

      // Browser-offline fast path: marked offline.
      self.navigator.onLine = false;
      const offlineHit = await get('/api/sessions');
      assert.equal(offlineHit.headers.get('X-Codoxear-Stale'), 'offline');
      self.navigator.onLine = true;

      // An available cached answer stays immediate while the permitted retry
      // is still pending; its eventual success restores normal live delivery.
      let releaseFetch;
      network = () => new Promise(resolve => { releaseFetch = resolve; });
      await advance(2000);
      const revalidating = dispatch('fetch', {request: request('/api/sessions')});
      assert.equal((await revalidating.response).headers.get('X-Codoxear-Stale'), 'suspect');
      releaseFetch(new Response('fresh'));
      await revalidating.finish();
      network = async () => new Response('fresh');
      const recovered = await get('/api/sessions');
      assert.equal(recovered.headers.get('X-Codoxear-Stale'), null);
      assert.equal(await recovered.text(), 'fresh');
    """)


def test_sse_post_and_cross_origin_requests_are_pure_passthrough() -> None:
    run_worker("""
      for (const req of [
        request('/api/sessions/s1/live?after=c1'),
        request('/api/sessions/s1/send', { method: 'POST' }),
        request('/static/app.css', { method: 'HEAD' }),
        request('https://other.example/static/app.css?v=1'),
      ]) {
        const event = dispatch('fetch', { request: req });
        assert.equal(event.response, undefined);
        assert.equal(event.waits.length, 0);
      }
      assert.equal(fetches.length, 0);
      assert.equal(puts, 0);
      // JSON polling is not the excluded SSE stream.
      await get('/api/sessions/s1/messages/live?after=c1');
      assert.ok(await cached('codoxear-api-v1', '/api/sessions/s1/messages/live?after=c1'));
    """)


def test_non_200_responses_never_replace_or_create_cached_entries() -> None:
    run_worker("""
      for (const [path, options, name] of [
        ['/', { mode: 'navigate' }, 'codoxear-html-v1'],
        ['/api/me', {}, 'codoxear-api-v1'],
        ['/static/icon.png', {}, 'codoxear-static-v1'],
      ]) {
        network = async () => new Response('known-good');
        await get(path, options);
        for (const status of [304, 401, 500]) {
          network = async () => new Response(status === 304 ? null : 'error', { status });
          await get(path, options);
          assert.equal(await (await cached(name, path)).text(), 'known-good');
        }
      }
      network = async () => new Response(null, { status: 304 });
      assert.equal((await get('/static/app.js?v=bad')).status, 304);
      assert.equal(await cached('codoxear-static-v1', '/static/app.js?v=bad'), undefined);
      network = async () => new Response('unauthorized', { status: 401 });
      await get('/api/uncached');
      offline();
      await assert.rejects(get('/api/uncached'), /offline/);
    """)


def test_versioned_static_cache_first_and_prunes_other_versions_of_same_path() -> None:
    run_worker("""
      await get('/static/app.js?v=old');
      await get('/static/other.js?v=old');
      const before = fetches.length;
      offline();
      assert.equal(await (await get('/static/app.js?v=old')).text(), 'online');
      assert.equal(fetches.length, before);
      network = async () => new Response('new');
      await get('/static/app.js?v=new');
      assert.equal(await cached('codoxear-static-v1', '/static/app.js?v=old'), undefined);
      assert.ok(await cached('codoxear-static-v1', '/static/other.js?v=old'));
      assert.equal(await (await cached('codoxear-static-v1', '/static/app.js?v=new')).text(), 'new');
      network = async () => new Response('broken', { status: 500 });
      await get('/static/app.js?v=failed');
      assert.ok(await cached('codoxear-static-v1', '/static/app.js?v=new'));
      assert.equal(await cached('codoxear-static-v1', '/static/app.js?v=failed'), undefined);
    """)


def test_unversioned_static_serves_cache_before_background_revalidation_finishes() -> None:
    run_worker("""
      await get('/static/monaco/editor.worker.js');
      let resolveFetch;
      network = () => new Promise(resolve => { resolveFetch = resolve; });
      const event = dispatch('fetch', { request: request('/static/monaco/editor.worker.js') });
      assert.equal(event.waits.length, 1);
      assert.equal(await (await event.response).text(), 'online');
      resolveFetch(new Response('updated worker'));
      await event.finish();
      assert.equal(await (await cached('codoxear-static-v1', '/static/monaco/editor.worker.js')).text(), 'updated worker');
      offline();
      assert.equal(await (await get('/static/monaco/editor.worker.js')).text(), 'updated worker');
      await assert.rejects(get('/uncached-icon.png'), /offline/);
    """)


def test_install_activate_and_push_notification_behavior() -> None:
    run_worker("""
      await caches.open('codoxear-static-old');
      await caches.open('codoxear-sw-meta-v1');
      await caches.open('other-app');
      await caches.open('codoxear-api-v1');
      await dispatch('install').finish();
      offline();
      await dispatch('activate').finish();
      assert.equal(skips, 1);
      assert.deepEqual(clientCalls, ['claim']);
      assert.equal((await caches.keys()).includes('codoxear-static-old'), false);
      assert.ok((await caches.keys()).includes('codoxear-sw-meta-v1'));
      assert.ok((await caches.keys()).includes('other-app'));
      assert.ok(warnings.length > 0);
      await dispatch('push', { data: { json: () => ({ session_display_name: 'Repo', notification_text: 'Done', session_id: 'a b' }) } }).finish();
      assert.equal(notifications[0].title, 'Repo');
      assert.equal(notifications[0].options.body, 'Done');
      assert.equal(notifications[0].options.data.url, origin + '/#session=a%20b');
      await dispatch('push', { data: { json() { throw new Error('malformed'); } } }).finish();
      assert.equal(notifications[1].title, 'Codoxear');
      assert.equal(notifications[1].options.body, 'New final response');
      let closed = false;
      await dispatch('notificationclick', { notification: { data: { url: origin + '/#session=s1' }, close() { closed = true; } } }).finish();
      assert.equal(closed, true);
      assert.equal(clientCalls[1], origin + '/#session=s1');
    """)


def test_prefixed_scope_routes_api_and_sse_without_capturing_other_app_paths() -> None:
    run_worker("""
      self.registration.scope = origin + '/codoxear/';
      await get('/codoxear/api/sessions');
      offline();
      assert.equal(await (await get('/codoxear/api/sessions')).text(), 'online');
      assert.ok(await cached('codoxear-api-v1', '/codoxear/api/sessions'));
      assert.equal(await cached('codoxear-static-v1', '/codoxear/api/sessions'), undefined);
      const before = fetches.length;
      for (const path of ['/codoxear/api/sessions/s1/live?after=c1', '/api/sessions', '/elsewhere/app.js?v=1']) {
        const event = dispatch('fetch', { request: request(path) });
        assert.equal(event.response, undefined);
        assert.equal(event.waits.length, 0);
      }
      assert.equal(fetches.length, before);
      network = async () => new Response('prefixed shell');
      await get('/codoxear/?entry=1', { mode: 'navigate' });
      offline();
      assert.equal(await (await get('/codoxear/?entry=2', { mode: 'navigate' })).text(), 'prefixed shell');
    """)


def test_activation_warms_shell_assets_and_authenticated_apis_before_any_reload() -> None:
    families = [path.stem for path in (WORKER.parent / 'themes').glob('*.css')]
    run_worker('const themeFamilies = ' + json.dumps(families) + ';' + r"""
      for (const scope of [origin + '/', origin + '/codoxear/']) {
        stores.clear(); fetches.length = 0;
        self.registration.scope = scope;
        const shell = `<link href="app.css?v=release">
          <script src='dist/app.bundle.js?v=release'></script>
          <link href="manifest.webmanifest?v=release">
          <link href="favicon.svg?size=32&amp;v=release">
          <link href="apple-touch-icon.png?v=release">
          <link href="app.css?v=release">
          <link href="https://other.example/foreign.css?v=release">
          <script src="unversioned.js"></script>
          <script>document.write('<link href="themes/' + family + '.css?v=release">')</script>`;
        const expectedAssets = ['app.css?v=release', 'dist/app.bundle.js?v=release',
          'manifest.webmanifest?v=release', 'favicon.svg?size=32&v=release',
          'apple-touch-icon.png?v=release', ...themeFamilies.map(f => `themes/${f}.css?v=release`)];
        network = async req => {
          assert.equal(req.credentials, 'same-origin');
          return new Response(req.url === scope ? shell : req.url);
        };
        const event = dispatch('activate');
        assert.equal(clientCalls.at(-1), 'claim', 'claim starts immediately');
        await event.finish();
        assert.equal(await (await cached('codoxear-html-v1', scope)).text(), shell);
        for (const path of expectedAssets) {
          const url = new URL(path, scope).href;
          assert.equal(await (await cached('codoxear-static-v1', url)).text(), url);
          assert.equal(fetches.filter(u => u === url).length, 1, 'deduplicate assets');
        }
        for (const path of ['api/me', 'api/sessions']) {
          const url = new URL(path, scope).href;
          assert.equal((await cached('codoxear-api-v1', url)).status, 200);
        }
        assert.equal(fetches.length, 3 + expectedAssets.length);
        offline();
        assert.equal(await (await get(scope + '?entry=offline', {mode: 'navigate'})).text(), shell);
        assert.equal((await get(new URL('api/sessions', scope).href)).status, 200);
      }
    """)


def test_offline_activation_claims_and_prunes_without_rejection() -> None:
    run_worker("""
      await caches.open('codoxear-html-old');
      await caches.open('another-app');
      offline();
      const event = dispatch('activate');
      assert.deepEqual(clientCalls, ['claim']);
      await event.finish();
      assert.equal((await caches.keys()).includes('codoxear-html-old'), false);
      assert.ok((await caches.keys()).includes('another-app'));
      assert.equal(puts, 0);
      assert.equal(warnings.length, 3);
    """)


def test_activation_warmup_skips_non_200_and_isolates_asset_failures() -> None:
    run_worker("""
      const shell = '<link href="app.css?v=new"><script src="broken.js?v=new"></script>';
      network = async req => {
        if (req.url.endsWith('/')) return new Response(shell);
        if (req.url.includes('broken.js')) throw new TypeError('asset unavailable');
        if (req.url.endsWith('/api/me')) return new Response('unauthenticated', {status: 401});
        if (req.url.endsWith('/api/sessions')) return new Response(null, {status: 304});
        return new Response('asset');
      };
      await dispatch('activate').finish();
      assert.equal(await cached('codoxear-api-v1', '/api/me'), undefined);
      assert.equal(await cached('codoxear-api-v1', '/api/sessions'), undefined);
      assert.equal(await cached('codoxear-static-v1', '/broken.js?v=new'), undefined);
      assert.ok(await cached('codoxear-static-v1', '/app.css?v=new'));
      assert.ok(await cached('codoxear-static-v1', '/themes/clay.css?v=new'));
      assert.equal(warnings.length, 1);
    """)


def test_network_first_hanging_fetch_times_out_but_late_success_refreshes_cache() -> None:
    run_worker("""

      await get('/api/me');
      let resolveFetch;
      network = () => new Promise(resolve => { resolveFetch = resolve; });
      const event = dispatch('fetch', {request: request('/api/me')});
      await flush();
      await advance(3000);
      assert.equal((await event.response).headers.get('X-Codoxear-Stale'), 'timeout');
      let finished = false;
      event.finish().then(() => { finished = true; });
      await flush();
      assert.equal(finished, false);
      resolveFetch(new Response('late success'));
      await event.finish();
      assert.equal(await (await cached('codoxear-api-v1', '/api/me')).text(), 'late success');
      assert.equal(timers.size, 0);
      // Slow completed reads cool down per key, without being labelled unavailable.
      const before = fetches.length;
      const slowHit = await get('/api/me');
      assert.equal(slowHit.headers.get('X-Codoxear-Stale'), 'timeout');
      assert.equal(fetches.length, before);

    """)


def test_derivative_and_range_requests_bypass_interception_at_root_and_prefix() -> None:
    run_worker("""
      for (const scope of [origin + '/', origin + '/codoxear/']) {
        self.registration.scope = scope;
        const base = new URL(scope).pathname;
        const derivativePaths = [
          '/api/sessions/s1/file/office_preview?path=deck.pptx',
          '/api/sessions/s1/file/video_preview?path=clip.mkv',
          '/api/files/office_preview?path=/tmp/deck.pptx',
          '/api/files/video_preview?path=/tmp/clip.mkv',
        ];
        for (const path of derivativePaths) {
          const event = dispatch('fetch', { request: request(base.slice(0, -1) + path) });
          assert.equal(event.response, undefined, path);
          assert.equal(event.waits.length, 0, path);
        }
        const blobPath = base + 'api/sessions/s1/file/blob?path=large.bin';
        network = async () => new Response('complete bytes');
        await get(blobPath);
        assert.ok(await cached('codoxear-api-v1', blobPath), 'non-Range blob caching stays intact');
        const before = fetches.length;
        for (const path of [blobPath, base + 'api/sessions', base + 'static/app.js?v=1']) {
          const event = dispatch('fetch', { request: request(path, { headers: { Range: 'bytes=0-0' } }) });
          assert.equal(event.response, undefined, 'Range must not receive cached full 200: ' + path);
          assert.equal(event.waits.length, 0);
        }
        assert.equal(fetches.length, before, 'bypassed requests are left to the browser');
        network = async () => new Response('partial', { status: 206, headers: { 'Content-Range': 'bytes 0-6/14' } });
        const partial = await get(blobPath + '&partial=1');
        assert.equal(partial.status, 206);
        assert.equal(await cached('codoxear-api-v1', blobPath + '&partial=1'), undefined, '206 is never stored');
      }
      assert.equal(puts > 0, true, 'control requests did populate caches');
    """)


def test_network_first_delivers_headers_before_gated_cache_put_and_finish_owns_storage() -> None:
    run_worker("""
      const openCache = caches.open;
      let releasePut, putStarted = false;
      caches.open = async (name) => {
        const cache = await openCache(name);
        if (name !== 'codoxear-api-v1') return cache;
        return { ...cache, async put(request, response) {
          putStarted = true;
          await new Promise(resolve => { releasePut = resolve; });
          return cache.put(request, response);
        } };
      };
      network = async () => new Response('large response body');
      const event = dispatch('fetch', { request: request('/api/sessions') });
      assert.equal(event.waits.length, 1, 'waitUntil is registered synchronously');
      let delivered = false;
      event.response.then(() => { delivered = true; });
      await flush();
      assert.equal(delivered, true, 'headers do not wait for body storage');
      assert.equal(putStarted, true);
      const response = await event.response;
      assert.equal(response.headers.get('X-Codoxear-Stale'), null);
      assert.equal(await response.text(), 'large response body', 'delivered body is independent of its clone');
      let finished = false;
      event.finish().then(() => { finished = true; });
      await flush();
      assert.equal(finished, false, 'event lifetime covers the gated cache write');
      assert.equal(await cached('codoxear-api-v1', '/api/sessions'), undefined);
      releasePut();
      await event.finish();
      assert.equal(await (await cached('codoxear-api-v1', '/api/sessions')).text(), 'large response body');
      assert.equal(timers.size, 0);
    """)


def test_network_first_cache_write_failures_do_not_reject_live_responses() -> None:
    run_worker("""
      const openCache = caches.open;
      for (const failureKind of ['throw', 'reject']) {
        caches.open = async (name) => {
          const cache = await openCache(name);
          if (name !== 'codoxear-api-v1') return cache;
          return { ...cache, put() {
            if (failureKind === 'throw') throw new Error('quota');
            return Promise.reject(new Error('quota'));
          } };
        };
        network = async () => new Response('live ' + failureKind);
        const event = dispatch('fetch', { request: request('/api/cache-failure-' + failureKind) });
        const response = await event.response;
        assert.equal(response.status, 200);
        assert.equal(await response.text(), 'live ' + failureKind);
        await event.finish();
        assert.equal(response.headers.get('X-Codoxear-Stale'), null);
      }
    """)


def test_versioned_static_delivery_still_awaits_cache_storage() -> None:
    run_worker("""
      const openCache = caches.open;
      let releasePut;
      caches.open = async (name) => {
        const cache = await openCache(name);
        if (name !== 'codoxear-static-v1') return cache;
        return { ...cache, async put(request, response) {
          await new Promise(resolve => { releasePut = resolve; });
          return cache.put(request, response);
        } };
      };
      network = async () => new Response('static body');
      const event = dispatch('fetch', { request: request('/static/app.js?v=1') });
      let delivered = false;
      event.response.then(() => { delivered = true; });
      await flush();
      assert.equal(delivered, false, 'static policy keeps its awaited write');
      releasePut();
      assert.equal(await (await event.response).text(), 'static body');
      await event.finish();
      assert.equal(await (await cached('codoxear-static-v1', '/static/app.js?v=1')).text(), 'static body');
    """)


def test_network_first_offline_fast_path_skips_fetch_but_misses_try_network() -> None:
    run_worker("""
      for (const [path, options, name] of [
        ['/?warm=1', {mode: 'navigate'}, 'codoxear-html-v1'],
        ['/api/sessions?cached=1', {}, 'codoxear-api-v1'],
      ]) {
        self.navigator.onLine = true;
        network = async () => new Response('cached');
        await get(path, options);
        self.navigator.onLine = false;
        const before = fetches.length;
        network = () => { throw new Error('fast path must not fetch'); };
        const target = options.mode === 'navigate' ? '/?offline=1' : path;
        assert.equal(await (await get(target, options)).text(), 'cached');
        assert.equal(fetches.length, before);
        assert.equal(timers.size, 0);
        network = async () => new Response('network despite offline hint');
        const miss = options.mode === 'navigate' ? '/uncached' : '/api/sessions?uncached=1';
        assert.equal(await (await get(miss, options)).text(), 'network despite offline hint');
        assert.equal(fetches.length, before + 1);
        assert.equal(await (await cached(name, miss)).text(), 'network despite offline hint');
        assert.equal(timers.size, 0);
      }
    """)


def test_network_first_quick_success_clears_timeout_and_missing_navigator_works() -> None:
    run_worker("""
      delete self.navigator;
      for (const [path, options, name] of [
        ['/', {mode: 'navigate'}, 'codoxear-html-v1'],
        ['/api/me', {}, 'codoxear-api-v1'],
      ]) {
        network = async () => new Response('stale');
        await get(path, options);
        network = async () => new Response('fresh');
        assert.equal(await (await get(path, options)).text(), 'fresh');
        assert.equal(await (await cached(name, path)).text(), 'fresh');
        assert.equal(timers.size, 0);
        await advance(3000);
        assert.equal(await (await cached(name, path)).text(), 'fresh');
      }
    """)


def test_uncached_slow_request_keeps_running_until_retained_deadline() -> None:
    run_worker("""

      let resolveFetch, delivered = false;
      network = () => new Promise(resolve => { resolveFetch = resolve; });
      const event = dispatch('fetch', {request: request('/api/uncached')});
      event.response.then(() => { delivered = true; });
      await flush(); await advance(10000);
      assert.equal(delivered, false, 'a cache delivery budget is not an uncached deadline');
      resolveFetch(new Response('late'));
      assert.equal(await (await event.response).text(), 'late');
      await event.finish();
      assert.equal(await (await cached('codoxear-api-v1', '/api/uncached')).text(), 'late');

    """)


def test_failed_refresh_backoff_is_per_key_and_suppresses_repeat_fetches() -> None:
    run_worker("""

      await get('/api/me');
      offline();
      assert.equal((await get('/api/me')).headers.get('X-Codoxear-Stale'), 'suspect');
      const before = fetches.length;
      for (let i = 0; i < 10; i++) {
        assert.equal((await get('/api/me')).headers.get('X-Codoxear-Stale'), 'suspect');
      }
      assert.equal(fetches.length, before, 'failed refresh cooldown suppresses repeated fetches');
      assert.equal(timers.size, 0);
      network = async () => new Response('independent');
      assert.equal(await (await get('/api/other')).text(), 'independent');

    """)


def test_completed_background_response_of_any_status_restores_normal_delivery() -> None:
    run_worker("""

      for (const status of [200, 304, 401, 500]) {
        const path = '/api/status-' + status;
        await get(path);
        offline(); await get(path);
        await advance(2000);
        let releaseFetch;
        network = () => new Promise(resolve => { releaseFetch = resolve; });
        const revalidation = dispatch('fetch', {request: request(path)});
        assert.equal((await revalidation.response).headers.get('X-Codoxear-Stale'), 'suspect');
        releaseFetch(new Response(status === 304 ? null : 'alive', {status}));
        await revalidation.finish();
        network = async () => new Response('fresh');
        const recovered = await get(path);
        assert.equal(recovered.headers.get('X-Codoxear-Stale'), null);
        assert.equal(await recovered.text(), 'fresh');
      }

    """)


def test_backoff_expiry_starts_one_shared_background_refresh() -> None:
    run_worker("""

      await get('/api/me'); offline(); await get('/api/me');
      const before = fetches.length;
      await advance(1999); await get('/api/me');
      assert.equal(fetches.length, before);
      let resolveFetch;
      network = () => new Promise(resolve => { resolveFetch = resolve; });
      await advance(1);
      const a = dispatch('fetch', {request: request('/api/me')});
      const b = dispatch('fetch', {request: request('/api/me')});
      assert.equal((await a.response).headers.get('X-Codoxear-Stale'), 'suspect');
      assert.equal((await b.response).headers.get('X-Codoxear-Stale'), 'suspect');
      assert.equal(fetches.length, before + 1, 'concurrent stale hits share a retained refresh');
      resolveFetch(new Response('recovered'));
      await Promise.all([a.finish(), b.finish()]);

    """)


def test_cache_miss_is_independent_of_another_keys_backoff() -> None:
    run_worker("""

      await get('/api/me'); offline(); await get('/api/me');
      network = async () => new Response('miss');
      const fresh = await get('/api/uncached');
      assert.equal(fresh.headers.get('X-Codoxear-Stale'), null);
      assert.equal(await fresh.text(), 'miss');
      const hit = await get('/api/me');
      assert.equal(hit.headers.get('X-Codoxear-Stale'), 'suspect', 'another resource success cannot make old bytes fresh');

    """)


def test_repeated_failed_background_refreshes_back_off() -> None:
    run_worker("""

      await get('/api/me'); offline(); await get('/api/me');
      const first = fetches.length;
      await advance(2000); await get('/api/me');
      assert.equal(fetches.length, first + 1);
      await advance(3999); await get('/api/me');
      assert.equal(fetches.length, first + 1);
      await advance(1); await get('/api/me');
      assert.equal(fetches.length, first + 2, 'repeated failures exponentially back off');

    """)


def test_slow_connection_timeout_never_latches_and_the_bound_adapts() -> None:
    run_worker("""

      await get('/api/me');
      let resolveFetch;
      network = () => new Promise(resolve => { resolveFetch = resolve; });
      const slow = dispatch('fetch', {request: request('/api/me')});
      await flush(); await advance(3000);
      assert.equal((await slow.response).headers.get('X-Codoxear-Stale'), 'timeout');
      await advance(1000); resolveFetch(new Response('late'));
      await slow.finish();
      // A different key uses the learned 8s bound and is not blocked by backoff.
      let resolveOther, delivered = false;
      network = () => new Promise(resolve => { resolveOther = resolve; });
      const other = dispatch('fetch', {request: request('/api/other')});
      other.response.then(() => { delivered = true; });
      await flush(); await advance(5000);
      assert.equal(delivered, false);
      resolveOther(new Response('slow live'));
      assert.equal((await other.response).headers.get('X-Codoxear-Stale'), null);
      await other.finish();

    """)


def test_elapsed_silence_never_becomes_unreachability_and_read_is_bounded() -> None:
    run_worker("""

      await get('/api/me');
      let retainedSignal;
      network = (_request, options) => { retainedSignal = options.signal; return new Promise(() => {}); };
      const events = [];
      for (let i = 0; i < 6; i++) {
        const event = dispatch('fetch', {request: request('/api/me')}); events.push(event);
        await flush(); await advance(3000);
        assert.equal((await event.response).headers.get('X-Codoxear-Stale'), 'timeout');
      }
      assert.equal(fetches.length, 2, 'all stale hits share the same pending fetch');
      await advance(60000);
      await Promise.all(events.map(e => e.finish()));
      assert.equal(retainedSignal.aborted, true);
      assert.equal((await get('/api/me')).headers.get('X-Codoxear-Stale'), 'timeout');

    """)


def test_browser_offline_hint_skips_refresh_and_marks_cached_navigation() -> None:
    run_worker("""

      await get('/', {mode: 'navigate'});
      self.navigator.onLine = false;
      const before = fetches.length;
      const event = dispatch('fetch', {request: request('/?offline=1', {mode: 'navigate'})});
      assert.equal((await event.response).headers.get('X-Codoxear-Stale'), 'offline');
      await event.finish();
      assert.equal(fetches.length, before);
      const marker = await (await cached('codoxear-sw-meta-v1', '/sw-meta/last-offline-navigation')).json();
      assert.equal(marker.offline, true);

    """)
