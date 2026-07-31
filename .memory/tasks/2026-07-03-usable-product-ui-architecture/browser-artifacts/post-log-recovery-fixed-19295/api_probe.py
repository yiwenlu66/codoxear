#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

BASE = sys.argv[1].rstrip('/')
PASSWORD = sys.argv[2]
RECOVERY_SID = "post-log-recovery-fixed"
CONTROL_SID = "post-log-completed-control"
LARGE_SID = "post-log-large-cursor"
SENTINEL = "POST_LOG_BOUND_DEATH_SENTINEL"
STOPPED = "The backend process stopped before completing this turn."
FIRST = "FIRST_EVENT_SENTINEL"

jar = CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def request(method: str, path: str, body: dict | None = None) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with opener.open(req, timeout=15) as resp:
            raw = resp.read().decode()
            return {"status": resp.status, "payload": json.loads(raw) if raw else None}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw
        return {"status": e.code, "payload": payload}


def get(path: str) -> dict:
    return request('GET', path)


def post(path: str, body: dict) -> dict:
    return request('POST', path, body)


def session_by_id(payload: dict, sid: str) -> dict | None:
    for row in payload.get('sessions', []):
        if row.get('session_id') == sid:
            return row
    return None

login = post('/api/login', {'password': PASSWORD})
initial_sessions = get('/api/sessions')
recovery_row = session_by_id(initial_sessions['payload'], RECOVERY_SID)
control_row = session_by_id(initial_sessions['payload'], CONTROL_SID)
large_row = session_by_id(initial_sessions['payload'], LARGE_SID)

tail = get(f'/api/sessions/{RECOVERY_SID}/messages/tail?limit=80')
history = get(f'/api/sessions/{RECOVERY_SID}/messages/history?limit=80')
live = get(f'/api/sessions/{RECOVERY_SID}/messages/live')
search_prompt = get(f'/api/sessions/{RECOVERY_SID}/messages/search?' + urllib.parse.urlencode({'q': SENTINEL, 'limit': 10, 'count_max': 100, 'text_max': 500}))
search_error = get(f'/api/sessions/{RECOVERY_SID}/messages/search?' + urllib.parse.urlencode({'q': STOPPED, 'limit': 10, 'count_max': 100, 'text_max': 500}))
export = get(f'/api/sessions/{RECOVERY_SID}/messages/export')
unknown_control_tail = get(f'/api/sessions/{CONTROL_SID}/messages/tail?limit=80')
send = post(f'/api/sessions/{RECOVERY_SID}/send', {'text': 'should be blocked'})
enqueue = post(f'/api/sessions/{RECOVERY_SID}/enqueue', {'text': 'should be blocked'})
attach = post(f'/api/sessions/{RECOVERY_SID}/inject_file', {'filename': 'blocked.txt', 'attachment_index': 1, 'data_b64': 'YmxvY2tlZA=='})
unattended = get(f'/api/sessions/{RECOVERY_SID}/unattended')
large_tail = get(f'/api/sessions/{LARGE_SID}/messages/tail?limit=80')
large_event_cursor = None
if large_tail['status'] == 200:
    for ev in large_tail['payload'].get('events', []):
        if ev.get('codoxear_lifecycle') == 'backend_stopped_after_log_bind':
            large_event_cursor = ev.get('history_cursor')
            break
large_history = get(f'/api/sessions/{LARGE_SID}/messages/history?' + urllib.parse.urlencode({'cursor': large_event_cursor or '', 'limit': 20})) if large_event_cursor else {'status': None, 'payload': None}
large_search_first = get(f'/api/sessions/{LARGE_SID}/messages/search?' + urllib.parse.urlencode({'q': FIRST, 'limit': 10, 'count_max': 100, 'text_max': 500}))
large_search_stopped = get(f'/api/sessions/{LARGE_SID}/messages/search?' + urllib.parse.urlencode({'q': STOPPED, 'limit': 10, 'count_max': 100, 'text_max': 500}))
rediscovered_sessions = get('/api/sessions')
rediscovered_tail = get(f'/api/sessions/{RECOVERY_SID}/messages/tail?limit=80')

out = {
    'login': login,
    'initial_sessions': initial_sessions,
    'recovery_row': recovery_row,
    'control_row': control_row,
    'large_row': large_row,
    'tail': tail,
    'history': history,
    'live': live,
    'search_prompt': search_prompt,
    'search_error': search_error,
    'export': export,
    'unknown_control_tail': unknown_control_tail,
    'send': send,
    'enqueue': enqueue,
    'attach': attach,
    'unattended': unattended,
    'large_tail': large_tail,
    'large_event_cursor_present': bool(large_event_cursor),
    'large_history': large_history,
    'large_search_first': large_search_first,
    'large_search_stopped': large_search_stopped,
    'rediscovered_sessions': rediscovered_sessions,
    'rediscovered_tail': rediscovered_tail,
}
print(json.dumps(out, indent=2, sort_keys=True))
