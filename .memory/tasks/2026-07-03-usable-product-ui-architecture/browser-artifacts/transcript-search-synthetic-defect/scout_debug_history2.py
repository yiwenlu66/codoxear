"""Debug 2: force silent turn fully into an older history page; also test
the split case (user in older window, close in current history window)."""
from __future__ import annotations
import json, tempfile
from pathlib import Path
from codoxear.rollout_log import _read_chat_tail_page, _read_chat_history_page
from codoxear.rollout_chat_events import _NO_RESPONSE_TEXT

def write_log(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

def build(rows_count_after):
    p = Path(tempfile.mkdtemp()) / "log.jsonl"
    rows = []
    ts = [1.0]
    def add_ans(label):
        rows.append({"type":"event_msg","ts":ts[0],"payload":{"type":"user_message","message":f"{label} q"}})
        rows.append({"type":"event_msg","ts":ts[0]+0.5,"payload":{"type":"agent_message","phase":"final_answer","message":f"{label} a"}})
        rows.append({"type":"event_msg","ts":ts[0]+1.0,"payload":{"type":"task_complete","turn_id":label}})
        ts[0] += 2.0
    for i in range(5): add_ans(f"pre{i}")
    rows.append({"type":"event_msg","ts":ts[0],"payload":{"type":"user_message","message":"silent user prompt"}})
    rows.append({"type":"event_msg","ts":ts[0]+1.0,"payload":{"type":"task_complete","turn_id":"silent","last_agent_message":None}})
    ts[0]+=2.0
    for i in range(rows_count_after): add_ans(f"post{i}")
    write_log(p, rows)
    return p

for after in (30, 50):
    print(f"--- after={after} ---")
    p = build(after)
    events, before, after_byte, has_older = _read_chat_tail_page(p, limit=80)
    tail_has_nr = any(e.get("text")==_NO_RESPONSE_TEXT for e in events)
    print(f"  TAIL: {len(events)} events has_older={has_older} tail_has_no_response={tail_has_nr}")
    cur = before
    pg = 0
    while True:
        he, nxt, ho = _read_chat_history_page(p, before_byte=cur, limit=60)
        pg += 1
        has_nr = any(e.get("text")==_NO_RESPONSE_TEXT for e in he)
        has_user = any(e.get("text")=="silent user prompt" for e in he)
        # detect close row presence by scanning raw? we infer: if user in window and no close, open turn
        print(f"  HIST p{pg}: {len(he)} ev has_older={ho} has_no_response={has_nr} has_silent_user={has_user}")
        if not ho: break
        cur = nxt
        if pg > 8: break
