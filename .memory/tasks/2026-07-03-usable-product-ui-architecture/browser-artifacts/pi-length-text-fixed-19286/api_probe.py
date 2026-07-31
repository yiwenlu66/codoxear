#!/usr/bin/env python3
"""Real-server API proof for fixed Pi visible-text length false-idle."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.request import HTTPCookieProcessor, build_opener

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:19286"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/pi-length-text-api")
PASSWORD = sys.argv[3] if len(sys.argv) > 3 else "test-password"
OUT.mkdir(parents=True, exist_ok=True)
NO_RESPONSE_TEXT = "The backend completed this turn without producing a response."
EXPECTED = {
    "pi-length-text-prefix-fixed": {"roles": ["user", "assistant"], "busy": True, "partial_class": "narration", "partial_search": True, "continuation": False},
    "pi-length-text-continuation-fixed": {"roles": ["user", "assistant", "assistant"], "busy": True, "partial_class": "narration", "partial_search": True, "continuation": True},
    "pi-stop-text-control": {"roles": ["user", "assistant"], "busy": False, "partial_class": "final_response", "partial_search": False, "continuation": False},
}

jar = CookieJar()
opener = build_opener(HTTPCookieProcessor(jar))


def request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with opener.open(req, timeout=15) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw.decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            parsed = {"raw": raw.decode("utf-8", errors="replace")}
        return exc.code, parsed


def get(path: str) -> tuple[int, dict]:
    return request("GET", path)


def save(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


status, me = get("/api/me")
save("00-me-before-login.json", {"status": status, "body": me})
assert status == 401, (status, me)
status, login = request("POST", "/api/login", {"password": PASSWORD})
save("01-login.json", {"status": status, "body": login})
assert status == 200, (status, login)
status, sessions = get("/api/sessions")
save("02-sessions.json", {"status": status, "body": sessions})
assert status == 200, sessions
rows = sessions.get("sessions", [])
by_id = {r.get("id") or r.get("session_id"): r for r in rows}
missing = sorted(set(EXPECTED) - set(by_id))
assert not missing, (missing, sorted(by_id))


def tail(session_id: str) -> dict:
    status, body = get(f"/api/sessions/{session_id}/messages/tail?limit=80")
    save(f"tail-{session_id}.json", {"status": status, "body": body})
    assert status == 200, body
    return body


def search(session_id: str, query: str, label: str) -> dict:
    q = urllib.parse.urlencode({"q": query, "limit": "10"})
    status, body = get(f"/api/sessions/{session_id}/messages/search?{q}")
    save(f"search-{session_id}-{label}.json", {"status": status, "body": body})
    assert status == 200, body
    return body

summary = {"base": BASE, "scenarios": {}, "session_rows": by_id}
for sid, spec in EXPECTED.items():
    t = tail(sid)
    events = t.get("events", [])
    roles = [e.get("role") for e in events]
    assistant = [e for e in events if e.get("role") == "assistant"]
    row = by_id[sid]
    completed = search(sid, "completed this turn", "completed")
    partial_search = search(sid, "partial before compaction", "partial")
    continuation_search = search(sid, "resuming after compaction", "continuation")
    record = {
        "tail_roles": roles,
        "tail_events": events,
        "session_busy": row.get("busy"),
        "assistant_classes": [e.get("message_class") for e in assistant],
        "search_completed_match_count": completed.get("match_count", 0),
        "search_partial_match_count": partial_search.get("match_count", 0),
        "search_continuation_match_count": continuation_search.get("match_count", 0),
        "has_no_response_text": any(NO_RESPONSE_TEXT in (e.get("text") or "") for e in events),
        "row": row,
    }
    assert roles == spec["roles"], (sid, roles, events)
    assert row.get("busy") is spec["busy"], (sid, row)
    assert assistant and assistant[0].get("message_class") == spec["partial_class"], (sid, assistant)
    assert record["has_no_response_text"] is False, (sid, events)
    assert completed.get("match_count", 0) == 0, (sid, completed)
    if spec["partial_search"]:
        assert partial_search.get("match_count", 0) >= 1, (sid, partial_search)
    else:
        assert partial_search.get("match_count", 0) == 0, (sid, partial_search)
    if spec["continuation"]:
        assert continuation_search.get("match_count", 0) >= 1, (sid, continuation_search)
        assert assistant[-1].get("message_class") == "narration", (sid, assistant)
    else:
        assert continuation_search.get("match_count", 0) == 0, (sid, continuation_search)
    summary["scenarios"][sid] = record
save("SUMMARY.json", summary)
print(json.dumps(summary, indent=2, sort_keys=True, default=str))
print("\nALL ASSERTIONS PASSED")
