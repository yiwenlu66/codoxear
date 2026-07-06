#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from http.cookiejar import CookieJar
from urllib.request import HTTPCookieProcessor, build_opener

NO_RESPONSE = "The backend completed this turn without producing a response."
BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:19242"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/codoxear-docker-sandbox-19242/artifacts/api-search")
OUT.mkdir(parents=True, exist_ok=True)

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
        with opener.open(req, timeout=10) as resp:
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
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

status, me = get("/api/me")
save("00-me-before-login.json", {"status": status, "body": me})
status, login = request("POST", "/api/login", {"password": "test-password"})
save("01-login.json", {"status": status, "body": login})
status, sessions = get("/api/sessions")
save("02-sessions.json", {"status": status, "body": sessions})
assert status == 200, sessions
ids = {row.get("id") or row.get("session_id") for row in sessions.get("sessions", [])}
required = {"search-codex-noresp", "search-codex-answered", "search-cc-noresp", "search-cc-apierr"}
assert required <= ids, sorted(required - ids)


def search(session_id: str, query: str) -> dict:
    status, body = get(f"/api/sessions/{session_id}/messages/search?" + urllib.parse.urlencode({"q": query, "limit": "5"}))
    save(f"search-{session_id}-{query[:18].replace(' ', '_').replace('/', '_')}.json", {"status": status, "body": body})
    assert status == 200, body
    return body

def history(session_id: str, cursor: str, label: str) -> dict:
    status, body = get(f"/api/sessions/{session_id}/messages/history?" + urllib.parse.urlencode({"cursor": cursor, "limit": "20"}))
    save(f"history-{session_id}-{label}.json", {"status": status, "body": body})
    assert status == 200, body
    return body

codex_no = search("search-codex-noresp", "backend completed this turn without producing a response")
assert codex_no["match_count"] == 1, codex_no
assert codex_no["matches"][0]["text"] == NO_RESPONSE, codex_no
assert codex_no["matches"][0].get("history_cursor"), codex_no
assert codex_no["matches"][0].get("load_cursor"), codex_no
codex_hist = history("search-codex-noresp", codex_no["matches"][0]["load_cursor"], "load-cursor")
assert any(e.get("text") == NO_RESPONSE for e in codex_hist.get("events", [])), codex_hist

cc_no = search("search-cc-noresp", "backend completed this turn without producing a response")
assert cc_no["match_count"] == 1, cc_no
assert cc_no["matches"][0]["text"] == NO_RESPONSE, cc_no
assert cc_no["matches"][0].get("history_cursor"), cc_no
assert cc_no["matches"][0].get("load_cursor"), cc_no
cc_hist = history("search-cc-noresp", cc_no["matches"][0]["load_cursor"], "load-cursor")
assert any(e.get("text") == NO_RESPONSE for e in cc_hist.get("events", [])), cc_hist

answered_no = search("search-codex-answered", "backend completed this turn without producing a response")
assert answered_no["match_count"] == 0, answered_no
answer_real = search("search-codex-answered", "CODEX-ANSWER-SEARCH")
assert answer_real["match_count"] == 1 and answer_real["matches"][0]["text"] == "CODEX-ANSWER-SEARCH", answer_real
apierr = search("search-cc-apierr", "503 Search Proof")
assert apierr["match_count"] == 1 and "503 Search Proof" in apierr["matches"][0]["text"], apierr
assert apierr["matches"][0]["text"] != NO_RESPONSE, apierr

summary = {
    "base": BASE,
    "sessions": sorted(required),
    "codex_no_response_match_count": codex_no["match_count"],
    "codex_no_response_history_has_row": any(e.get("text") == NO_RESPONSE for e in codex_hist.get("events", [])),
    "cc_no_response_match_count": cc_no["match_count"],
    "cc_no_response_history_has_row": any(e.get("text") == NO_RESPONSE for e in cc_hist.get("events", [])),
    "answered_no_response_match_count": answered_no["match_count"],
    "answered_real_match_count": answer_real["match_count"],
    "cc_api_error_match_text": apierr["matches"][0]["text"],
}
save("SUMMARY.json", summary)
print(json.dumps(summary, indent=2, sort_keys=True))
