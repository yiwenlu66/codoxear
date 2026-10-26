#!/usr/bin/env python3
"""Real browser file chooser + real HTTP fault verification in an owned Docker.

Run via docker_upload_robustness_verify.sh. All API traffic uses its Docker-only
proxy port. The fake broker is a log/read fixture; no provider credentials or
prompt submission are needed. Assertions use production UI and HTTP resources.
"""
import argparse
import hashlib
import http.cookiejar
import json
from pathlib import Path
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, required=True)
p.add_argument('--container', required=True)
p.add_argument('--artifacts', required=True)
p.add_argument('--session', required=True)
a = p.parse_args()
assert a.port != 8743
base = f'http://127.0.0.1:{a.port}'
art = Path(a.artifacts)
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
report = {}


def http(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=30) as r:
        return json.loads(r.read())


def ab(*args):
    r = subprocess.run(['agent-browser', '--session', a.session, *args], capture_output=True, text=True, timeout=45)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return r.stdout


def js(code):
    data = json.loads(ab('eval', code, '--json'))
    return data['data']['result']


def wait(fn, seconds=25):
    until = time.monotonic() + seconds
    value = None
    while time.monotonic() < until:
        value = fn()
        if value:
            return value
        time.sleep(.4)
    raise AssertionError(f'timed out: last={value}')


def chips():
    return js('Array.from(document.querySelectorAll("#stagedAttachments .stagedAttachmentChip")).map(e=>({text:e.textContent,pending:e.dataset.pending||null}))')


def resource():
    return http('/api/sessions/activity-peek/attachments')


def check(name, value):
    report[name] = value
    (art / 'report.json').write_text(json.dumps(report, indent=2))
    print(name, json.dumps(value), flush=True)


def choose(name):
    path = art / name
    payload = bytes(range(256)) * 4096
    path.write_bytes(payload)
    ab('upload', '#imgInput', str(path))
    return hashlib.sha256(payload).hexdigest()


http('/api/login', {'password': 'upload-verify-password'})
ab('open', base)
ab('fill', '#pw', 'upload-verify-password')
ab('click', '#loginBtn')
wait(lambda: js('Boolean(document.querySelector("#sessions .session"))'))
ab('click', '#sessions .session')
wait(lambda: js('!document.querySelector("#attachBtn").disabled'))
wait(lambda: js('Boolean(navigator.serviceWorker.controller)'))
# Capture the pre-upload real catalog via proxy, then replay it after mutation.
http('/api/sessions')
digest = choose('normal.bin')
wait(lambda: len(resource()['attachments']) == 1)
wait(lambda: len(chips()) == 1 and not chips()[0]['pending'])
wire = http('/__fault')['uploads']
assert len(wire) == 1 and wire[0]['bytes'] == 1048576 and wire[0]['sha256'] == digest
assert wire[0]['type'] == 'application/octet-stream'
check('raw_filechooser', {'wire': wire, 'chips': chips(), 'resource': resource()})
http('/__fault', {'stale_catalog': True})
# Real HTTP stale catalog response is delivered through the real SW/API path.
js('fetch("/api/sessions",{cache:"reload"}).then(r=>r.json()).then(v=>v.sessions.length)')
time.sleep(3)
assert len(chips()) == 1
ab('reload')
wait(lambda: len(chips()) == 1)
check('stale_catalog_after_reload', {'chips': chips(), 'checkpoint': js('Object.keys(localStorage).filter(k=>k.includes("attachment"))')})
http('/__fault', {'stale_catalog': False})
http('/api/sessions/activity-peek/attachments/clear', {})
wait(lambda: len(chips()) == 0, 35)
check('newer_delete', resource())
# Lose the successful POST response at TCP after the real origin committed.
http('/__fault', {'lose_next': True})
choose('lost.bin')
wait(lambda: len(resource()['attachments']) == 1)
wait(lambda: len(chips()) == 1 and not chips()[0]['pending'])
fault = http('/__fault')
assert len(fault['uploads']) == 2, fault
assert fault['status'][-1]['cache_control'] == 'no-store', fault
check('lost_committed_response', {'wire': fault['uploads'][-1], 'status': fault['status'], 'chips': chips()})
http('/api/sessions/activity-peek/attachments/clear', {})
wait(lambda: not chips(), 35)
# Status failure preserves unknown outcome and the retained File for Retry.
http('/__fault', {'lose_next': True, 'fail_status': True})
choose('unknown.bin')
wait(lambda: any('Retry' in c['text'] and 'unknown' in c['text'] for c in chips()))
assert len(resource()['attachments']) == 1
before = len(http('/__fault')['uploads'])
check('unknown_retained', {'chips': chips(), 'upload_posts': before})
ab('screenshot', str(art / 'unknown.png'))
http('/__fault', {'fail_status': False})
ab('click', '.stagedAttachmentRetry')
wait(lambda: len(chips()) == 1 and not chips()[0]['pending'])
assert len(http('/__fault')['uploads']) == before
check('retry_reconciles_without_duplicate', {'chips': chips(), 'resource': resource()})
http('/api/sessions/activity-peek/attachments/clear', {})
wait(lambda: not chips(), 35)
http('/__fault', {'cut_next': True})
choose('partial.bin')
wait(lambda: len(chips()) == 1 and not chips()[0]['pending'])
fault = http('/__fault')
assert len(fault['uploads']) == before + 2, fault
assert fault['uploads'][-1]['id'] == fault['uploads'][-2]['id']
check('partial_body_retry', {'wire': fault['uploads'][-2:], 'resource': resource()})
# Slow responses still arrive; SW may serve cache without implying unreachability.
http('/__fault', {'delay': 5})
slow = js('Promise.all(Array.from({length:16},()=>fetch("/api/sessions").then(async r=>({stale:r.headers.get("X-Codoxear-Stale"),ok:(await r.json()).sessions.length>0})) ))')
assert all(r['ok'] for r in slow)
time.sleep(7)
fault = http('/__fault')
network = js('({hidden:document.querySelector("#networkStatus").hidden,title:document.querySelector("#networkStatus").title,onLine:navigator.onLine})')
assert network['hidden'] is True, network
assert fault['max_catalog'] <= 2, fault
check('slow_cache_bounded', {'responses': slow, 'network': network, 'max_catalog': fault['max_catalog']})
http('/__fault', {'delay': 0})
ab('screenshot', str(art / 'final.png'))
# Hash the actual server-side file rather than trusting response metadata.
server_files = subprocess.check_output(['docker', 'exec', a.container, 'sh', '-c', 'find "$HOME/.local/share/codoxear/uploads" -type f -exec sha256sum {} \\;'], text=True)
assert digest in server_files, server_files
mutations = subprocess.check_output(['docker', 'exec', a.container, 'sh', '-c', 'cat "$HOME/broker-mutations.jsonl" 2>/dev/null || true'], text=True)
assert not mutations, mutations
check('readback_and_no_prompt', {'server_hashes': server_files, 'broker_mutations': 0})
print('PASS all browser upload robustness scenarios', flush=True)
