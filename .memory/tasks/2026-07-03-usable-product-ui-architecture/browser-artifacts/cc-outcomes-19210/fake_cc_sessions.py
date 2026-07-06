#!/usr/bin/env python3
from __future__ import annotations
import json, os, socket, threading, time
from pathlib import Path
APP=Path(os.environ.get('HOME','/home/tester'))/'.local'/'share'/'codoxear'
SOCKS=APP/'socks'
LOGS=APP/'cc-logs'
SOCKS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
SESSION_ID='11111111-2222-3333-4444-555555555555'

def row_user(text):
    return {"type":"user","sessionId":SESSION_ID,"timestamp":"2026-07-04T00:00:00.000Z","cwd":"/workspace","message":{"role":"user","content":[{"type":"text","text":text}]}}

def row_answer(text):
    return {"type":"assistant","sessionId":SESSION_ID,"timestamp":"2026-07-04T00:00:01.000Z","message":{"role":"assistant","content":[{"type":"text","text":text}],"stop_reason":"end_turn"}}

def row_turn_duration():
    return {"type":"system","subtype":"turn_duration","sessionId":SESSION_ID,"timestamp":"2026-07-04T00:00:03.000Z","durationMs":1234}

def row_api_error(text):
    return {"type":"system","subtype":"api_error","sessionId":SESSION_ID,"timestamp":"2026-07-04T00:00:03.000Z","error":text,"retryAttempt":3,"maxRetries":3}

def write_log(sid, rows):
    path=LOGS/f'{sid}.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r)+'\n')
    return path

def serve(sock_path: Path):
    try: sock_path.unlink()
    except FileNotFoundError: pass
    srv=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path)); srv.listen(16); srv.settimeout(0.4)
    def handle(conn):
        try:
            line=conn.makefile('rb').readline()
            req=json.loads(line.decode()) if line else {}
            cmd=req.get('cmd') if isinstance(req,dict) else None
            if cmd=='state': resp={"busy":False,"queue_len":0,"interrupted_idle":False,"token":None}
            elif cmd=='tail': resp={"tail":""}
            else: resp={"ok":True}
            conn.sendall((json.dumps(resp)+'\n').encode())
        except Exception as exc:
            try: conn.sendall((json.dumps({"error":str(exc)})+'\n').encode())
            except Exception: pass
        finally:
            try: conn.close()
            except Exception: pass
    while True:
        try: conn,_=srv.accept()
        except socket.timeout: continue
        except OSError: break
        threading.Thread(target=handle,args=(conn,),daemon=True).start()

sessions={
    'cc-noresp': [row_user('silent cc turn'), row_turn_duration()],
    'cc-apierr': [row_user('fail cc turn'), row_api_error('API Error: 503 Service Unavailable')],
    'cc-normal': [row_user('normal cc turn'), row_answer('CC-ANSWER-OK'), row_turn_duration()],
}
for sid, rows in sessions.items():
    log=write_log(sid, rows)
    sock=SOCKS/f'{sid}.sock'
    sidecar={"agent_backend":"cc","session_id":sid,"codex_pid":999001,"broker_pid":999002,"cwd":"/workspace","log_path":str(log),"start_ts":100.0,"owner":"terminal","source":"cc-cert"}
    (SOCKS/f'{sid}.json').write_text(json.dumps(sidecar)+'\n', encoding='utf-8')
    threading.Thread(target=serve,args=(sock,),daemon=True).start()
print(json.dumps({"ready":True,"sessions":list(sessions),"logs":str(LOGS)}), flush=True)
while True: time.sleep(60)
