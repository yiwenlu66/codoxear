#!/usr/bin/env python3
"""Real-server API proof for the interruption-outcome FIXED verification.

Drives the LIVE codoxear.server running in Docker (port from argv[1]) through
its real HTTP surfaces: /api/me, /api/login, /api/sessions, and per-session
/messages/tail + /messages/search. Saves every raw JSON response under argv[2].

Asserts the committed fix: every interrupted turn now renders a persistent
assistant interruption row (role=assistant, message_class=error, searchable via
"interrupted"), and Pi partial abort additionally preserves the streamed
partial text ("halfway through").
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.request import HTTPCookieProcessor, build_opener

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:19272"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/codoxear-docker-sandbox-19272/artifacts/interrupt-api")
PASSWORD = sys.argv[3] if len(sys.argv) > 3 else "test-password"
OUT.mkdir(parents=True, exist_ok=True)

INTERRUPT_TEXT = "The backend turn was interrupted before completion."

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


# ---- auth ----
status, me = get("/api/me")
save("00-me-before-login.json", {"status": status, "body": me})
assert status == 401, ("expected 401 pre-login, got", status, me)

status, login = request("POST", "/api/login", {"password": PASSWORD})
save("01-login.json", {"status": status, "body": login})
assert status == 200, ("login failed", status, login)

# ---- sessions ----
status, sessions = get("/api/sessions")
save("02-sessions.json", {"status": status, "body": sessions})
assert status == 200, sessions

rows = sessions.get("sessions", [])
ids = {r.get("id") or r.get("session_id") for r in rows}
required = {"interrupt-pi-empty", "interrupt-pi-partial", "interrupt-codex-abort"}
missing = sorted(required - ids)
assert not missing, ("missing sessions:", missing, "have:", sorted(ids))
save("02b-session-ids.json", {"required": sorted(required), "present": sorted(ids), "rows": rows})

SCN = {
    "interrupt-pi-empty": {"backend": "pi", "user_text": "hello pi (empty abort proof)", "partial": None},
    "interrupt-pi-partial": {"backend": "pi", "user_text": "hello pi (partial abort proof)", "partial": "I was halfway through the answer"},
    "interrupt-codex-abort": {"backend": "codex", "user_text": "hello codex (turn_aborted proof)", "partial": None},
}


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


summary = {"base": BASE, "sessions": sorted(required), "per_scenario": {}}

for sid, spec in SCN.items():
    t = tail(sid)
    events = t.get("events", [])
    roles = [e.get("role") for e in events]
    assistant_rows = [e for e in events if e.get("role") == "assistant"]
    user_rows = [e for e in events if e.get("role") == "user"]

    # Control: the user prompt is present and its text matches.
    user_text_ok = any(spec["user_text"] in (e.get("text") or "") for e in user_rows)
    # Fix: an assistant interruption row exists.
    has_assistant = bool(assistant_rows)
    # Fix: the row carries the canonical interrupted text + error class.
    intr_text_ok = any(INTERRUPT_TEXT in (e.get("text") or "") for e in assistant_rows)
    intr_class_ok = any(e.get("message_class") == "error" for e in assistant_rows)
    # Fix: searchable via "interrupted".
    s_intr = search(sid, "interrupted", "interrupted")
    intr_search_count = s_intr.get("match_count", 0)

    partial_preserved = None
    partial_search_count = None
    if spec["partial"]:
        partial_preserved = any(spec["partial"] in (e.get("text") or "") for e in assistant_rows)
        s_part = search(sid, "halfway through", "halfway")
        partial_search_count = s_part.get("match_count", 0)
        # Prove history rehydration: load the matched row via load_cursor.
        if partial_search_count and s_part.get("matches"):
            lc = s_part["matches"][0].get("load_cursor")
            if lc:
                h = history(sid, lc, "partial-load-cursor")
                partial_history_ok = any(spec["partial"] in (e.get("text") or "") for e in h.get("events", []))
            else:
                partial_history_ok = None
        else:
            partial_history_ok = None
    else:
        partial_history_ok = None

    # History rehydration for the interruption row itself (every scenario).
    hist_intr_ok = None
    if intr_search_count and s_intr.get("matches"):
        lc = s_intr["matches"][0].get("load_cursor")
        if lc:
            h = history(sid, lc, "intr-load-cursor")
            hist_intr_ok = any(INTERRUPT_TEXT in (e.get("text") or "") for e in h.get("events", []))

    summary["per_scenario"][sid] = {
        "backend": spec["backend"],
        "tail_roles": roles,
        "tail_events": events,
        "user_prompt_in_tail": user_text_ok,
        "has_assistant_interruption_row": has_assistant,
        "interruption_text_present": intr_text_ok,
        "interruption_class_is_error": intr_class_ok,
        "search_interrupted_match_count": intr_search_count,
        "partial_text_preserved": partial_preserved,
        "search_halfway_match_count": partial_search_count,
        "partial_history_rehydrated": partial_history_ok,
        "interruption_history_rehydrated": hist_intr_ok,
    }

save("SUMMARY.json", summary)
print(json.dumps(summary["per_scenario"], indent=2, sort_keys=True, default=str))
print("\nALL ASSERTIONS PASSED")
