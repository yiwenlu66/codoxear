#!/usr/bin/env python3
"""Real-server API proof for Pi no-visible-response terminal rows."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.request import HTTPCookieProcessor, build_opener

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:19280"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/pi-no-text-api")
PASSWORD = sys.argv[3] if len(sys.argv) > 3 else "test-password"
OUT.mkdir(parents=True, exist_ok=True)

NO_RESPONSE_TEXT = "The backend completed this turn without producing a response."
TERMINAL = {
    "pi-no-text-stop-empty": "hello pi stop empty proof",
    "pi-no-text-end-turn-empty": "hello pi end turn empty proof",
    "pi-no-text-stop-thinking": "hello pi stop thinking proof",
}
CONTROLS = {
    "pi-nonterminal-thinking-control": "hello pi nonterminal thinking control",
    "pi-tool-use-control": "hello pi tool use control",
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
required = set(TERMINAL) | set(CONTROLS)
missing = sorted(required - set(by_id))
assert not missing, ("missing sessions", missing, "present", sorted(by_id))


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


def history(session_id: str, cursor: str, label: str) -> dict:
    q = urllib.parse.urlencode({"cursor": cursor, "limit": "30"})
    status, body = get(f"/api/sessions/{session_id}/messages/history?{q}")
    save(f"history-{session_id}-{label}.json", {"status": status, "body": body})
    assert status == 200, body
    return body


summary: dict = {"base": BASE, "terminal": {}, "controls": {}, "session_rows": by_id}

for sid, prompt in TERMINAL.items():
    t = tail(sid)
    events = t.get("events", [])
    roles = [e.get("role") for e in events]
    assistant_rows = [e for e in events if e.get("role") == "assistant"]
    row = by_id[sid]
    srch = search(sid, "completed this turn", "completed")
    hist_ok = None
    if srch.get("matches"):
        cursor = srch["matches"][0].get("load_cursor")
        if cursor:
            h = history(sid, cursor, "completed-load-cursor")
            hist_ok = any(NO_RESPONSE_TEXT in (e.get("text") or "") for e in h.get("events", []))
    record = {
        "tail_roles": roles,
        "tail_events": events,
        "user_prompt_in_tail": any(prompt in (e.get("text") or "") for e in events),
        "assistant_no_response_row": any(NO_RESPONSE_TEXT in (e.get("text") or "") for e in assistant_rows),
        "assistant_class_error": any(e.get("message_class") == "error" for e in assistant_rows),
        "search_completed_match_count": srch.get("match_count", 0),
        "history_rehydrated": hist_ok,
        "session_busy": row.get("busy"),
        "session_ready": row.get("ready"),
        "session_remote_ready": row.get("remote_ready"),
        "session_can_send": row.get("can_send"),
        "row": row,
    }
    assert roles == ["user", "assistant"], (sid, roles, events)
    assert record["assistant_no_response_row"], (sid, events)
    assert record["assistant_class_error"], (sid, events)
    assert record["search_completed_match_count"] >= 1, (sid, srch)
    assert record["history_rehydrated"] is True, (sid, record)
    assert row.get("busy") is False, (sid, row)
    summary["terminal"][sid] = record

for sid, prompt in CONTROLS.items():
    t = tail(sid)
    events = t.get("events", [])
    roles = [e.get("role") for e in events]
    row = by_id[sid]
    srch = search(sid, "completed this turn", "completed")
    record = {
        "tail_roles": roles,
        "tail_events": events,
        "user_prompt_in_tail": any(prompt in (e.get("text") or "") for e in events),
        "search_completed_match_count": srch.get("match_count", 0),
        "session_busy": row.get("busy"),
        "session_ready": row.get("ready"),
        "session_remote_ready": row.get("remote_ready"),
        "session_can_send": row.get("can_send"),
        "row": row,
    }
    assert roles == ["user"], (sid, roles, events)
    assert record["search_completed_match_count"] == 0, (sid, srch)
    summary["controls"][sid] = record

save("SUMMARY.json", summary)
print(json.dumps(summary, indent=2, sort_keys=True, default=str))
print("\nALL ASSERTIONS PASSED")
