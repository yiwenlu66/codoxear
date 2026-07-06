"""Debug: where does the Codex no-response synthetic row land across pages?"""
from __future__ import annotations
import json, tempfile
from pathlib import Path
from codoxear.rollout_log import _read_chat_tail_page, _read_chat_history_page
from codoxear.rollout_chat_events import _NO_RESPONSE_TEXT

def write_log(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "log.jsonl"
    rows = []
    ts = 1.0
    def add_ans(label):
        global ts
        rows.append({"type":"event_msg","ts":ts,"payload":{"type":"user_message","message":f"{label} q"}})
        rows.append({"type":"event_msg","ts":ts+0.5,"payload":{"type":"agent_message","phase":"final_answer","message":f"{label} a"}})
        rows.append({"type":"event_msg","ts":ts+1.0,"payload":{"type":"task_complete","turn_id":label}})
        ts += 2.0
    for i in range(40): add_ans(f"pre{i}")
    rows.append({"type":"event_msg","ts":ts,"payload":{"type":"user_message","message":"trigger silent backend"}})
    rows.append({"type":"event_msg","ts":ts+1.0,"payload":{"type":"task_complete","turn_id":"silent","last_agent_message":None}})
    ts += 2.0
    for i in range(40): add_ans(f"post{i}")
    write_log(p, rows)

    events, before, after, has_older = _read_chat_tail_page(p, limit=80)
    print("TAIL page events:", len(events), "has_older:", has_older, "before_byte:", before)
    print("  tail contains no-response?", any(e.get("text")==_NO_RESPONSE_TEXT for e in events))
    # Walk history pages
    cur = before
    page = 0
    found_in = None
    while True:
        hevents, nxt, ho = _read_chat_history_page(p, before_byte=cur, limit=60)
        page += 1
        texts = [(e.get("role"), e.get("text")) for e in hevents]
        has_nr = any(e.get("text")==_NO_RESPONSE_TEXT for e in hevents)
        has_user = any(e.get("text")=="trigger silent backend" for e in hevents)
        has_close = any("silent" in str(e) for e in hevents)
        print(f"HIST page {page}: {len(hevents)} events has_older={ho} has_no_response={has_nr} has_user_silent={has_user}")
        if has_nr:
            found_in = page
            # print neighbors
            for e in hevents:
                if e.get("text")==_NO_RESPONSE_TEXT:
                    print("   no-response event:", e)
        if not ho: break
        cur = nxt
        if page > 10: break
    print("FOUND no-response in history page:", found_in)
