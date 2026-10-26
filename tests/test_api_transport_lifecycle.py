"""Execute api() with real Response bodies and controllable request deadlines."""
import json
import subprocess
from frontend_module_loader import module_path

API = module_path("app_api.js")
NETWORK = module_path("app_network.js")


def run_api(body):
    source = "const API_SOURCE = " + json.dumps(API.read_text()) + ";\n"
    source += "const NETWORK_SOURCE = " + json.dumps(NETWORK.read_text()) + ";\n"
    source += r"""
const vm = require('node:vm'), assert = require('node:assert/strict');
let clock = 0, timerId = 0;
const timers = new Map(), requests = [], reports = [];
const setTimeout = (fn, delay) => {const id = ++timerId; timers.set(id, {fn, at: clock + delay}); return id;};
const clearTimeout = id => timers.delete(id);
async function flush() {for (let i = 0; i < 60; i++) await Promise.resolve(); await new Promise(setImmediate);}
async function advance(ms) {
  clock += ms;
  for (const [id, timer] of [...timers]) if (timer.at <= clock) {timers.delete(id); timer.fn();}
  await flush();
}
let network = async () => Response.json({ok: true});
const ctx = {
  console, AbortController, performance: {now: () => clock}, setTimeout, clearTimeout,
  window: {CodoxearUrls: {resolveAppUrl: path => 'https://example.test/prefix' + path}},
  fetch: (url, options) => {requests.push({url, options}); return network(url, options);},
};
vm.createContext(ctx); vm.runInContext(API_SOURCE, ctx); vm.runInContext(NETWORK_SOURCE, ctx);
const api = ctx.window.CodoxearApi.api;
ctx.window.CodoxearApi.setTransportReporter(kind => reports.push(kind));
(async () => {
"""
    source += body + "\nconsole.log('API_DONE');\n})().catch(e => {console.error(e); process.exit(1);});"
    result = subprocess.run(["node", "-e", source], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "API_DONE" in result.stdout, "scenario did not complete"


def test_raw_blob_headers_json_and_authoritative_status_contract():
    run_api(r"""
      const blob = new Blob(['file bytes']);
      await api('/api/sessions/s1/inject_file', {method: 'POST', rawBody: blob,
        headers: {'X-Codoxear-Upload-Id': 'id-1', 'X-Codoxear-Filename': '%E4%B8%AD.png'}});
      assert.equal(requests[0].url, 'https://example.test/prefix/api/sessions/s1/inject_file');
      assert.equal(requests[0].options.body, blob);
      assert.equal(requests[0].options.headers['Content-Type'], 'application/octet-stream');
      assert.equal(requests[0].options.headers['X-Codoxear-Upload-Id'], 'id-1');
      await api('/api/custom', {method: 'POST', rawBody: blob, headers: new Headers({'content-type': 'image/png'})});
      assert.equal(requests[1].options.headers['content-type'], 'image/png');
      assert.equal(requests[1].options.headers['Content-Type'], undefined);
      await api('/api/json', {method: 'POST', body: {value: 1}, headers: [['X-Token', 'present']]});
      assert.equal(requests[2].options.body, '{"value":1}');
      assert.equal(requests[2].options.headers['Content-Type'], 'application/json');
      assert.equal(requests[2].options.headers['X-Token'], 'present');
      await assert.rejects(api('/api/bad', {body: {}, rawBody: blob}), /mutually exclusive/);
      assert.equal(requests.length, 3);
      await api('/api/sessions/s1/attachments/upload-status?upload_id=id-1');
      assert.equal(requests[3].options.cache, 'no-store');
      await api('/api/sessions', {cache: 'no-store'});
      assert.equal(requests[4].options.cache, 'no-store');
      assert.equal(requests[4].options.headers['If-None-Match'], undefined);
    """)


def test_http_json_validation_and_body_failure_are_separate_evidence():
    run_api(r"""
      network = async () => Response.json({error: 'denied'}, {status: 401});
      await assert.rejects(api('/api/fail'), error => error.status === 401 && error.obj.error === 'denied');
      assert.deepEqual(reports.splice(0), ['live']);
      network = async () => new Response('invalid JSON');
      await assert.rejects(api('/api/invalid'), error => error.name === 'SyntaxError');
      assert.deepEqual(reports.splice(0), ['live']);
      let body;
      network = async () => new Response(new ReadableStream({start(c) {body = c;}}));
      const pending = api('/api/body-failure');
      const rejected = assert.rejects(pending, /body reset/);
      await flush(); assert.deepEqual(reports, [], 'headers alone are not successful JSON completion');
      body.error(new TypeError('body reset')); await rejected;
      assert.deepEqual(reports.splice(0), ['failed']);
      network = async () => {throw new TypeError('connection reset');};
      await assert.rejects(api('/api/fetch-failure'), /connection reset/);
      assert.deepEqual(reports.splice(0), ['failed']);
    """)


def test_deadline_caller_cancel_and_worker_capacity_never_report_offline():
    run_api(r"""
      let signal;
      network = (_url, opts) => {signal = opts.signal; return new Promise(() => {});};
      const explicit = assert.rejects(api('/api/slow', {timeoutMs: 250}), e => e.name === 'TimeoutError' && e.code === 'REQUEST_TIMEOUT');
      await flush(); await advance(250); await explicit;
      assert.equal(signal.aborted, true); assert.deepEqual(reports, []);
      const controller = new AbortController();
      network = async (_url, opts) => {signal = opts.signal; return new Response(new ReadableStream({}));};
      const cancelled = assert.rejects(api('/api/body-cancel', {signal: controller.signal}), e => e.name === 'AbortError');
      await flush(); controller.abort(); await cancelled;
      assert.equal(signal.aborted, true); assert.deepEqual(reports, []);
      for (const kind of ['deadline', 'capacity']) {
        network = async () => Response.json({error: kind}, {status: 504, headers: {'X-Codoxear-Transport': kind}});
        await assert.rejects(api('/api/local-budget'), e => e.name === (kind === 'deadline' ? 'TimeoutError' : 'ReadCapacityError'));
      }
      assert.deepEqual(reports, []); assert.equal(timers.size, 0);
    """)


def test_default_body_budget_starts_after_upload_headers_not_during_transfer():
    run_api(r"""
      let resolveHeaders, signal;
      network = (_url, opts) => {signal = opts.signal; return new Promise(resolve => {resolveHeaders = resolve;});};
      const upload = assert.rejects(api('/api/sessions/s1/inject_file', {method: 'POST', rawBody: new Blob(['bytes'])}),
        e => e.name === 'TimeoutError');
      await flush(); await advance(120000);
      assert.equal(signal.aborted, false); assert.equal(timers.size, 0, 'no default total upload deadline');
      resolveHeaders(new Response(new ReadableStream({}), {headers: {'Content-Type': 'application/json'}}));
      await flush(); assert.equal(timers.size, 1);
      await advance(59999); assert.equal(signal.aborted, false);
      await advance(1); await upload;
      assert.equal(signal.aborted, true); assert.deepEqual(reports, []);
    """)


def test_stale_timeout_facts_do_not_change_indicator_but_native_failures_do():
    run_api(r"""
      const attrs = new Map();
      const status = {hidden: true, getAttribute: k => attrs.get(k), hasAttribute: k => attrs.has(k),
        setAttribute: (k,v) => attrs.set(k,v), removeAttribute: k => attrs.delete(k)};
      const statusText = {textContent: ''};
      const net = ctx.window.CodoxearNetwork.createNetworkStatusController({status, statusText, navigatorLike: {onLine: true}});
      ctx.window.CodoxearApi.setTransportReporter(kind => {
        reports.push(kind);
        if (kind === 'live') net.reportSuccess();
        else if (kind === 'stale') net.reportStale();
        else net.reportFailure();
      });
      network = async () => Response.json({old: true}, {headers: {'X-Codoxear-Stale': 'timeout'}});
      for (let i = 0; i < 5; i++) {
        const value = await api('/api/sessions');
        assert.equal(ctx.window.CodoxearApi.apiResponseStale(value), true);
        assert.equal(status.hidden, true);
      }
      assert.deepEqual(reports, []);
      network = async () => {throw new TypeError('reset');};
      await api('/api/one').catch(() => {}); assert.equal(status.hidden, true);
      await api('/api/two').catch(() => {}); assert.equal(status.hidden, false);
      network = async () => Response.json({ok: true}); await api('/api/live');
      assert.equal(status.hidden, true);
    """)


def test_default_get_headers_deadline_is_finite_without_service_worker():
    run_api(r"""
      let signal;
      network = (_url, opts) => {signal = opts.signal; return new Promise(() => {});};
      const rejected = assert.rejects(api('/api/sessions/s1/attachments/upload-status?upload_id=id'),
        e => e.name === 'TimeoutError');
      await flush(); await advance(59999); assert.equal(signal.aborted, false);
      await advance(1); await rejected;
      assert.equal(signal.aborted, true); assert.deepEqual(reports, []);
    """)


def test_failed_cached_body_is_not_native_network_failure():
    run_api(r"""
      let body;
      network = async () => new Response(new ReadableStream({start(c) {body = c;}}),
        {headers: {'X-Codoxear-Stale': 'timeout'}});
      const rejected = assert.rejects(api('/api/cache-body'), /cache body corrupt/);
      await flush(); body.error(new TypeError('cache body corrupt')); await rejected;
      assert.deepEqual(reports, []);
    """)
