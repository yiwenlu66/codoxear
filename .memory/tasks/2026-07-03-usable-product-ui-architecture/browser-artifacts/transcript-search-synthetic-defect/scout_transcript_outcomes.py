"""Read-only scout probe (final): do search / history / tail / live preserve
the synthetic no-response (Codex + CC) and CC terminal api_error rows?

Run from repo root: python3 /tmp/scout_transcript_outcomes.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from codoxear.message_cursor import decode_message_cursor, encode_message_cursor
from codoxear.message_routes import MessageRouteDeps
from codoxear.message_routes import handle_messages_history
from codoxear.message_routes import handle_messages_live
from codoxear.message_routes import handle_messages_search
from codoxear.message_routes import handle_messages_tail
from codoxear.rollout_chat_events import _NO_RESPONSE_TEXT
from codoxear.rollout_log import _read_chat_history_page
from codoxear.rollout_log import _read_chat_tail_page
from codoxear.session_model import Session

SECRET = b"scout-secret"


def write_log(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_session(td: str, log_path: Path, backend: str = "codex") -> Session:
    return Session(
        session_id="s1", thread_id="thread-1", broker_pid=1, codex_pid=1,
        agent_backend=backend, owned=False, start_ts=0.0, cwd=td,
        log_path=log_path, sock_path=Path(td) / "s1.sock",
    )


class Mgr:
    def __init__(self, session): self._s = session
    def refresh_session_meta(self, _sid): return None
    def get_session(self, _sid): return self._s
    def mark_log_delta(self, *a, **k): return None
    def _attach_notification_texts(self, events): return events


def deps_for(session):
    responses = []
    def json_response(_h, status, payload): responses.append((status, payload))
    def enc(*, kind, session, pos): return encode_message_cursor(kind=kind, session=session, pos=pos, secret=SECRET)
    def dec(token, *, kind, session): return decode_message_cursor(token, kind=kind, session=session, secret=SECRET)
    def snap(_sid, _session, **_kw): return {}, False, 0, None
    return MessageRouteDeps(
        require_auth=lambda _h: True, json_response=json_response,
        launch_attempt_transcript_for_session_id=lambda _sid: None,
        transcript_export_max_bytes=10 * 1024 * 1024,
        transcript_search_max_line_bytes=64 * 1024,
        decode_message_cursor=dec, encode_message_cursor=enc,
        record_metric=lambda _n, _v: None, message_runtime_snapshot=snap,
    ), responses


class FakeHandler:
    def _unauthorized(self): pass


def call_search(session, query):
    d, responses = deps_for(session)
    handle_messages_search(FakeHandler(), session_id="s1", query=f"q={query}", manager=Mgr(session), deps=d)
    return responses[0]


def call_tail(session):
    d, responses = deps_for(session)
    handle_messages_tail(FakeHandler(), session_id="s1", query="limit=80", manager=Mgr(session), deps=d)
    return responses[0]


def walk_history(session):
    """Return all event texts across every older history page from the tail cursor."""
    d, responses = deps_for(session)
    handle_messages_tail(FakeHandler(), session_id="s1", query="limit=80", manager=Mgr(session), deps=d)
    tail = responses[0][1]
    out = []
    cur = tail.get("history_cursor")
    visited = 0
    while cur and visited < 25:
        d2, r2 = deps_for(session)
        handle_messages_history(FakeHandler(), session_id="s1", query=f"cursor={cur}&limit=60", manager=Mgr(session), deps=d2)
        body = r2[0][1]
        out.extend((e.get("role"), e.get("text")) for e in body.get("events", []))
        cur = body.get("history_cursor")
        visited += 1
    return out, visited


def event_texts(body):
    return [(e.get("role"), e.get("message_class"), e.get("text")) for e in body.get("events", [])]


def report(title, ok, detail):
    print(f"[{'PASS' if ok else 'DEFECT'}] {title}\n        {detail}\n")


def codex_no_response_turn_rows():
    rows = []
    ts = [1.0]
    def ans(label):
        rows.append({"type": "event_msg", "ts": ts[0], "payload": {"type": "user_message", "message": f"{label} q"}})
        rows.append({"type": "event_msg", "ts": ts[0]+0.5, "payload": {"type": "agent_message", "phase": "final_answer", "message": f"{label} a"}})
        rows.append({"type": "event_msg", "ts": ts[0]+1.0, "payload": {"type": "task_complete", "turn_id": label}})
        ts[0] += 2.0
    for i in range(5): ans(f"pre{i}")                # few pre turns
    rows.append({"type": "event_msg", "ts": ts[0], "payload": {"type": "user_message", "message": "trigger silent backend"}})
    rows.append({"type": "event_msg", "ts": ts[0]+1.0, "payload": {"type": "task_complete", "turn_id": "silent", "last_agent_message": None}})
    ts[0] += 2.0
    for i in range(50): ans(f"post{i}")              # push silent turn into older history
    return rows


def main():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        print("=== Codex no-response turn ===")
        log_codex = td_path / "codex.jsonl"
        write_log(log_codex, codex_no_response_turn_rows())
        s_codex = make_session(td, log_codex, backend="codex")

        st, tail = call_tail(s_codex)
        # Synthetic row lives in older history (not tail). Walk history:
        hist_texts, pages = walk_history(s_codex)
        hist_flat = [t for _r, t in hist_texts]
        report(
            "history pagination preserves Codex synthetic no-response row",
            _NO_RESPONSE_TEXT in hist_flat,
            f"pages_walked={pages}; no-response in history={_NO_RESPONSE_TEXT in hist_flat}",
        )

        ss, sbody = call_search(s_codex, "backend completed this turn without producing a response")
        match_texts = [m.get("text") for m in sbody.get("matches", [])]
        report(
            "search finds Codex synthetic no-response text",
            _NO_RESPONSE_TEXT in match_texts,
            f"match_count={sbody.get('match_count')} matches={match_texts}",
        )

        # Control: ordinary user prompt is searchable
        sc, scbody = call_search(s_codex, "trigger silent backend")
        ctrl = [m.get("text") for m in scbody.get("matches", [])]
        report(
            "search control: ordinary user prompt is found",
            any("trigger silent backend" in t for t in ctrl),
            f"match_count={scbody.get('match_count')} matches={ctrl}",
        )

        print("=== Claude Code no-response turn (system/turn_duration, no assistant) ===")
        log_cc = td_path / "cc.jsonl"
        write_log(log_cc, [
            {"type": "user", "timestamp": "2026-01-01T00:00:00Z", "message": {"role": "user", "content": [{"type": "text", "text": "cc silent backend prompt"}]}},
            {"type": "system", "timestamp": "2026-01-01T00:00:05Z", "subtype": "turn_duration", "duration_ms": 1000},
        ])
        s_cc = make_session(td, log_cc, backend="cc")
        cct, cctail = call_tail(s_cc)
        cc_tail_texts = [t[2] for t in event_texts(cctail)]
        report("tail includes CC synthetic no-response text", _NO_RESPONSE_TEXT in cc_tail_texts, f"events={event_texts(cctail)}")
        ccs, ccbody = call_search(s_cc, "backend completed this turn without producing a response")
        cc_match = [m.get("text") for m in ccbody.get("matches", [])]
        report("search finds CC synthetic no-response text", _NO_RESPONSE_TEXT in cc_match, f"match_count={ccbody.get('match_count')} matches={cc_match}")

        print("=== Claude Code terminal api_error (real error text from log row) ===")
        log_ccerr = td_path / "cc_err.jsonl"
        write_log(log_ccerr, [
            {"type": "user", "timestamp": "2026-01-01T00:00:00Z", "message": {"role": "user", "content": [{"type": "text", "text": "cc failing backend prompt"}]}},
            {"type": "system", "timestamp": "2026-01-01T00:00:09Z", "subtype": "api_error", "retryAttempt": 3, "maxRetries": 3, "error": "API Error: 503 Service Unavailable"},
        ])
        s_ccerr = make_session(td, log_ccerr, backend="cc")
        et, etail = call_tail(s_ccerr)
        et_texts = [t[2] for t in event_texts(etail)]
        report("tail includes CC terminal api_error text", any("503" in (t or "") for t in et_texts), f"events={event_texts(etail)}")
        es, ebody = call_search(s_ccerr, "503")
        e_match = [m.get("text") for m in ebody.get("matches", [])]
        report("search finds CC terminal api_error real text", any("503" in (t or "") for t in e_match), f"match_count={ebody.get('match_count')} matches={e_match}")

        print("=== Fresh-process re-read (same handlers, same disk log) ===")
        st2, tail2 = call_tail(s_cc)
        nr2 = any(e.get("text") == _NO_RESPONSE_TEXT for e in tail2.get("events", []))
        report("re-read from disk reproduces CC synthetic no-response", nr2, f"events={event_texts(tail2)}")


if __name__ == "__main__":
    main()
