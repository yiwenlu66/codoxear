"""Finite worker reads: body, cache, concurrency and authoritative verification."""
from test_service_worker_offline import run_worker


def test_json_body_hang_falls_back_without_claiming_outage_then_late_body_warms_cache():
    run_worker(r"""
      network = async () => Response.json({version: 1});
      await get('/api/sessions');
      let body;
      network = async () => new Response(new ReadableStream({start(c) {body = c;}}),
        {headers: {'Content-Type': 'application/json'}});
      const event = dispatch('fetch', {request: request('/api/sessions')});
      await flush(); await advance(3000);
      const fallback = await event.response;
      assert.equal(fallback.headers.get('X-Codoxear-Stale'), 'timeout');
      assert.deepEqual(await fallback.json(), {version: 1});
      const before = fetches.length;
      const hits = Array.from({length: 12}, () => dispatch('fetch', {request: request('/api/sessions')}));
      for (const hit of hits) assert.equal((await hit.response).headers.get('X-Codoxear-Stale'), 'timeout');
      assert.equal(fetches.length, before);
      body.enqueue(new TextEncoder().encode('{"version":2}')); body.close();
      await Promise.all([event.finish(), ...hits.map(e => e.finish())]);
      assert.deepEqual(await (await cached('codoxear-api-v1', '/api/sessions')).json(), {version: 2});
    """)


def test_body_rejection_uses_cached_data_and_shared_fetch_rejections_back_off():
    run_worker(r"""
      network = async () => Response.json({saved: true}); await get('/api/me');
      let body;
      network = async () => new Response(new ReadableStream({start(c) {body = c;}}),
        {headers: {'Content-Type': 'application/json'}});
      const events = Array.from({length: 10}, () => dispatch('fetch', {request: request('/api/me')}));
      await flush(); body.error(new TypeError('truncated response'));
      for (const event of events) {
        const response = await event.response;
        assert.equal(response.headers.get('X-Codoxear-Stale'), 'suspect');
        assert.deepEqual(await response.json(), {saved: true}); await event.finish();
      }
      assert.equal(fetches.length, 2);
      const before = fetches.length;
      await get('/api/me'); assert.equal(fetches.length, before);
    """)


def test_uncached_full_body_deadline_has_explicit_non_outage_metadata():
    run_worker(r"""
      let signal;
      network = async (_req, opts) => {
        signal = opts.signal;
        return new Response(new ReadableStream({}), {headers: {'Content-Type': 'application/json'}});
      };
      const event = dispatch('fetch', {request: request('/api/no-cache')});
      let delivered = false; event.response.then(() => {delivered = true;});
      await flush(); await advance(3000);
      assert.equal(delivered, false, 'headers are not usable JSON');
      await advance(57000);
      const response = await event.response;
      assert.equal(response.status, 504);
      assert.equal(response.headers.get('X-Codoxear-Transport'), 'deadline');
      assert.equal(signal.aborted, true); await event.finish();
    """)


def test_cancelling_one_consumer_does_not_cancel_shared_retained_read():
    run_worker(r"""
      let resolveFetch, retainedSignal;
      network = (_req, opts) => { retainedSignal = opts.signal; return new Promise(resolve => {resolveFetch = resolve;}); };
      const controller = new AbortController();
      const first = dispatch('fetch', {request: request('/api/me', {signal: controller.signal})});
      const second = dispatch('fetch', {request: request('/api/me')});
      await flush(); controller.abort();
      assert.equal(retainedSignal.aborted, false);
      assert.equal(fetches.length, 1);
      resolveFetch(Response.json({ok: true}));
      assert.deepEqual(await (await second.response).json(), {ok: true});
      assert.deepEqual(await (await first.response).json(), {ok: true});
      await Promise.all([first.finish(), second.finish()]);
    """)


def test_retained_read_pool_bounds_never_settling_work_without_serializing_foreground():
    run_worker(r"""
      const seeds = Array.from({length: 5}, (_, i) => '/api/low-' + i);
      for (const path of seeds) await get(path);
      offline(); for (const path of seeds) await get(path);
      await advance(2000);
      network = () => new Promise(() => {});
      const events = seeds.map(path => dispatch('fetch', {request: request(path)}));
      for (const event of events) assert.equal((await event.response).headers.get('X-Codoxear-Stale'), 'suspect');
      assert.equal(fetches.length, 15, 'incoming demands use three background slots plus two permitted foreground slots');
      network = async () => Response.json({foreground: true});
      assert.deepEqual(await (await get('/api/foreground')).json(), {foreground: true});
      network = () => new Promise(() => {});
      const blockers = Array.from({length: 8}, (_, i) => dispatch('fetch', {request: request('/api/block-' + i)}));
      await flush();
      assert.equal(fetches.length, 19, 'five retained retries plus three additional foreground slots');
      for (const event of blockers.slice(3)) assert.equal((await event.response).headers.get('X-Codoxear-Transport'), 'capacity');
      await advance(60000);
      await Promise.all([...events, ...blockers].map(e => e.finish()));
      const before = fetches.length;
      const saturated = await get('/api/still-bounded');
      assert.equal(saturated.headers.get('X-Codoxear-Transport'), 'capacity');
      assert.equal(fetches.length, before, 'unabortable work keeps its slot, not an ever-growing orphan chain');
    """)


def test_cache_faults_have_separate_budget_and_never_delay_live_delivery():
    run_worker(r"""
      const open = caches.open;
      let writes = 0;
      caches.open = async name => {
        const cache = await open(name);
        return {...cache, put() { writes++; return new Promise(() => {}); }};
      };
      network = async () => Response.json({live: true});
      const events = Array.from({length: 8}, (_, i) => dispatch('fetch', {request: request('/api/cache-' + i)}));
      for (const event of events) assert.deepEqual(await (await event.response).json(), {live: true});
      assert.equal(writes, 4, 'stuck cache puts cannot accumulate');
      await advance(5000);
      await Promise.all(events.map(e => e.finish()));
      assert.deepEqual(await (await get('/api/after-cache-budget')).json(), {live: true});
      assert.equal(writes, 4);
      caches.open = async () => { throw new Error('cache unavailable'); };
      assert.deepEqual(await (await get('/api/cache-open-fault')).json(), {live: true});
    """)


def test_no_store_and_upload_status_are_native_even_with_seeded_cache_and_prefix():
    run_worker(r"""
      for (const scope of [origin + '/', origin + '/codoxear/']) {
        self.registration.scope = scope;
        const base = new URL(scope).pathname;
        for (const [path, options] of [
          [base + 'api/sessions/s1/attachments/upload-status?upload_id=id', {}],
          [base + 'api/me', {cache: 'no-store'}],
          [base + 'api/sessions/s1/inject_file', {method: 'POST'}],
        ]) {
          await (await caches.open('codoxear-api-v1')).put(request(path), Response.json({old: true}));
          const event = dispatch('fetch', {request: request(path, options)});
          assert.equal(event.response, undefined); assert.equal(event.waits.length, 0);
        }
      }
      assert.equal(fetches.length, 0);
    """)


def test_conditional_and_unconditional_readers_never_share_a_304_response():
    run_worker(r"""
      const releases = new Map();
      network = req => new Promise(resolve => {releases.set(req.headers.get('If-None-Match') || 'unconditional', resolve);});
      const conditional = dispatch('fetch', {request: request('/api/sessions', {headers: {'If-None-Match': 'v1'}})});
      const unconditional = dispatch('fetch', {request: request('/api/sessions')});
      const equivalent = dispatch('fetch', {request: request('/api/sessions', {headers: {'if-none-match': 'v1'}})});
      await flush();
      assert.equal(fetches.length, 2, 'conditional headers are part of request equivalence');
      releases.get('v1')(new Response(null, {status: 304}));
      releases.get('unconditional')(Response.json({sessions: ['fresh']}));
      assert.equal((await conditional.response).status, 304);
      assert.equal((await equivalent.response).status, 304);
      const response = await unconditional.response;
      assert.equal(response.status, 200);
      assert.deepEqual(await response.json(), {sessions: ['fresh']});
      await Promise.all([conditional.finish(), unconditional.finish(), equivalent.finish()]);
      assert.deepEqual(await (await cached('codoxear-api-v1', '/api/sessions')).json(), {sessions: ['fresh']});
    """)


def test_expired_cooldown_retries_live_before_a_stalled_cache_lookup():
    run_worker(r"""
      network = async () => Response.json({version: 1});
      await get('/api/me');
      offline();
      assert.equal((await get('/api/me')).headers.get('X-Codoxear-Stale'), 'suspect');
      await advance(2001);
      const open = caches.open;
      let lookups = 0;
      caches.open = async name => {
        const cache = await open(name);
        return {...cache, match() { lookups++; return new Promise(() => {}); }};
      };
      let releaseFetch;
      network = () => new Promise(resolve => { releaseFetch = resolve; });
      const before = fetches.length;
      const first = dispatch('fetch', {request: request('/api/me')});
      const second = dispatch('fetch', {request: request('/api/me')});
      await flush();
      assert.equal(fetches.length, before + 1, 'expired cooldown starts one shared retry without waiting for Cache Storage');
      assert.equal(lookups, 1, 'equivalent callers share the stalled lookup too');
      let delivered = false;
      first.response.then(() => {delivered = true;});
      releaseFetch(Response.json({version: 2}));
      await flush();
      assert.equal(delivered, true, 'a complete healthy answer is delivered without advancing the cache budget clock');
      for (const event of [first, second]) {
        const response = await event.response;
        assert.equal(response.headers.get('X-Codoxear-Stale'), null);
        assert.deepEqual(await response.json(), {version: 2});
        await event.finish();
      }
      assert.equal(clock, 2001, 'cache match never gates retry or healthy delivery');
    """)


def test_expired_retry_uses_free_foreground_capacity_when_background_pool_is_full():
    run_worker(r"""
      const paths = Array.from({length: 4}, (_, i) => '/api/saturated-' + i);
      network = async () => Response.json({cached: true});
      for (const path of paths) await get(path);
      offline(); for (const path of paths) await get(path);
      await advance(2001);
      const releases = new Map();
      network = req => new Promise(resolve => {releases.set(req.url, resolve);});
      const background = paths.slice(0, 3).map(path => dispatch('fetch', {request: request(path)}));
      for (const event of background) assert.equal((await event.response).headers.get('X-Codoxear-Stale'), 'suspect');
      assert.equal(releases.size, 3);
      const open = caches.open;
      caches.open = async name => {
        const cache = await open(name);
        return {...cache, match(req, options) {
          return req.url === origin + paths[3] ? new Promise(() => {}) : cache.match(req, options);
        }};
      };
      const before = fetches.length;
      const foreground = dispatch('fetch', {request: request(paths[3])});
      const equivalent = dispatch('fetch', {request: request(paths[3])});
      await flush();
      assert.equal(fetches.length, before + 1, 'three background reads must not block a permitted incoming foreground retry');
      releases.get(origin + paths[3])(Response.json({live: true}));
      for (const event of [foreground, equivalent]) {
        const response = await event.response;
        assert.equal(response.headers.get('X-Codoxear-Stale'), null);
        assert.deepEqual(await response.json(), {live: true}); await event.finish();
      }
      assert.equal(clock, 2001, 'stalled cache lookup never delays foreground delivery');
      // Three background reads still occupy slots. Five further foreground
      // reads fill the global pool; further incoming requests cannot exceed eight.
      const blockers = Array.from({length: 5}, (_, i) => dispatch('fetch', {request: request('/api/full-' + i)}));
      await flush();
      assert.equal(releases.size, 9, 'four prior fetches plus five additional pending foreground reads');
      const pendingCount = fetches.length;
      offline();
      const saturated = await get('/api/no-capacity');
      assert.equal(saturated.headers.get('X-Codoxear-Transport'), 'capacity');
      assert.equal(fetches.length, pendingCount, 'global pool capacity remains eight regardless of background admission');
      await advance(60000);
      await Promise.all([...background, ...blockers].map(event => event.finish()));
    """)


def test_expired_retry_cannot_exceed_a_full_global_read_pool():
    run_worker(r"""
      const path = '/api/expired-at-capacity';
      network = async () => Response.json({cached: true}); await get(path);
      offline(); await get(path); await advance(2001);
      network = () => new Promise(() => {});
      const blockers = Array.from({length: 8}, (_, i) => dispatch('fetch', {request: request('/api/occupied-' + i)}));
      await flush();
      const open = caches.open;
      caches.open = async name => {
        const cache = await open(name);
        return {...cache, match(req, options) {
          return req.url === origin + path ? new Promise(() => {}) : cache.match(req, options);
        }};
      };
      const before = fetches.length;
      const expired = dispatch('fetch', {request: request(path)});
      await flush();
      assert.equal(fetches.length, before, 'expired demand does not bypass the eight-read global limit');
      await advance(5000);
      const response = await expired.response;
      assert.equal(response.headers.get('X-Codoxear-Transport'), 'capacity');
      assert.equal(fetches.length, before);
      await expired.finish(); await advance(55000);
      await Promise.all(blockers.map(event => event.finish()));
    """)



def test_completion_queries_are_native_so_aborted_keystrokes_never_consume_worker_slots():
    run_worker(r"""
      for (const scope of [origin + '/', origin + '/codoxear/']) {
        self.registration.scope = scope;
        const path = new URL(scope).pathname + 'api/sessions/s1/file/complete';
        const controllers = Array.from({length: 8}, () => new AbortController());
        const events = controllers.map((controller, i) => dispatch('fetch', {request:
          new Request(origin + path + '?path=' + i, {signal: controller.signal})}));
        controllers.forEach(controller => controller.abort());
        for (const event of events) {
          assert.equal(event.response, undefined, 'browser owns native fetch and cancellation');
          assert.equal(event.waits.length, 0, 'obsolete queries retain no worker lifetime');
        }
        const latest = dispatch('fetch', {request: new Request(origin + path + '?path=latest')});
        assert.equal(latest.response, undefined);
        network = async () => Response.json({healthy: true});
        assert.deepEqual(await (await get(new URL('api/me', scope).pathname)).json(), {healthy: true});
      }
      assert.equal(fetches.length, 2, 'only independent JSON reads used the worker pool');
    """)
