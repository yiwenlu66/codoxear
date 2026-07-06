#!/usr/bin/env python3
"""Real-server API proof for Pi length follow-up boundary."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.request import HTTPCookieProcessor, build_opener

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:19283"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/pi-length-api")
PASSWORD = sys.argv[3] if len(sys.argv) > 3 else "test-password"
OUT.mkdir(parents=True, exist_ok=True)
NO_RESPONSE_TEXT = "The backend completed this turn without producing a response."

EXPECTED = {
    "pi-stop-empty-regression": {
        "prompt": "hello pi stop empty regression",
        "expected_roles": ["user", "assistant"],
        "expect_no_response": True,
        "expect_busy": False,
    },
    "pi-length-prefix-control": {
        "prompt": "hello pi length prefix control",
        "expected_roles": ["user"],
        "expect_no_response": False,
        "expect_busy": True,
    },
    "pi-length-continuation-control": {
        "prompt": "hello pi length continuation control",
        "expected_roles": ["user", "assistant"],
        "expect_no_response": False,
        "expect_busy": True,
        "expect_continuation_text": "continuing with a tool",
        "expect_continuation_class": "narration",
    },
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
assert status == 401, ("expected 401 pre-login", status, me)
status, login = request("POST", "/api/login", {"password": PASSWORD})
save("01-login.json", {"status": status, "body": login})
assert status == 200, ("login failed", status, login)
status, sessions = get("/api/sessions")
save("02-sessions.json", {"status": status, "body": sessions})
assert status == 200, sessions
rows = sessions.get("sessions", [])
by_id = {r.get("id") or r.get("session_id"): r for r in rows}
missing = sorted(set(EXPECTED) - set(by_id))
assert not missing, ("missing sessions", missing, sorted(by_id))


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


summary: dict = {"base": BASE, "scenarios": {}, "session_rows": by_id}

for sid, spec in EXPECTED.items():
    t = tail(sid)
    events = t.get("events", [])
    roles = [e.get("role") for e in events]
    row = by_id[sid]
    completed = search(sid, "completed this turn", "completed")
    continuation = search(sid, "continuing with a tool", "continuation")
    no_response_rows = [e for e in events if NO_RESPONSE_TEXT in (e.get("text") or "")]
    continuation_rows = [e for e in events if spec.get("expect_continuation_text", "\u0000") in (e.get("text") or "")]
    record = {
        "tail_roles": roles,
        "tail_events": events,
        "session_busy": row.get("busy"),
        "search_completed_match_count": completed.get("match_count", 0),
        "search_continuation_match_count": continuation.get("match_count", 0),
        "has_no_response_row": bool(no_response_rows),
        "continuation_rows": continuation_rows,
        "row": row,
    }
    assert roles == spec["expected_roles"], (sid, roles, events)
    assert bool(no_response_rows) is spec["expect_no_response"], (sid, events)
    assert row.get("busy") is spec["expect_busy"], (sid, row)
    if spec["expect_no_response"]:
        assert completed.get("match_count", 0) >= 1, (sid, completed)
    else:
        assert completed.get("match_count", 0) == 0, (sid, completed)
    if "expect_continuation_text" in spec:
        assert continuation_rows, (sid, events)
        assert continuation_rows[0].get("message_class") == spec["expect_continuation_class"], (sid, continuation_rows)
        assert continuation.get("match_count", 0) >= 1, (sid, continuation)
    summary["scenarios"][sid] = record

save("SUMMARY.json", summary)
print(json.dumps(summary, indent=2, sort_keys=True, default=str))
print("\nALL ASSERTIONS PASSED")
