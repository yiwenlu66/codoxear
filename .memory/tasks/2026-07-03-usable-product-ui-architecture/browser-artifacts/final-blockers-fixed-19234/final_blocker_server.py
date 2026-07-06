#!/usr/bin/env python3
from __future__ import annotations
import json, os, socket, subprocess, sys, threading, time
from pathlib import Path
APP=Path(os.environ.get('HOME','/home/tester'))/'.local'/'share'/'codoxear'
SOCKS=APP/'socks'; LOGS=APP/'logs'
SOCKS.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
PORT=int(os.environ['CODEX_WEB_PORT'])

def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row)+'\n')

def cc_user(sid,text): return {"type":"user","sessionId":sid,"timestamp":"2026-07-04T00:00:00.000Z","cwd":"/workspace","message":{"role":"user","content":[{"type":"text","text":text}]}}
def codex_interrupted_rows(): return [
    {"type":"session_meta","payload":{"id":"fresh-interrupt","source":"cli"}},
    {"type":"event_msg","payload":{"type":"user_message","message":"first"},"ts":10.0},
    {"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"working"}]},"ts":11.0},
]

def serve(sock_path: Path, state: dict):
    try: sock_path.unlink()
    except FileNotFoundError: pass
    srv=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(str(sock_path)); srv.listen(16); srv.settimeout(0.25)
    def handle(conn):
        try:
            line=conn.makefile('rb').readline(); req=json.loads(line.decode()) if line else {}; cmd=req.get('cmd') if isinstance(req,dict) else None
            resp=dict(state) if cmd=='state' else ({"tail":""} if cmd=='tail' else {"ok":True})
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

def register(sid, backend, rows, state):
    log=LOGS/f'{sid}.jsonl'; write_jsonl(log, rows)
    sock=SOCKS/f'{sid}.sock'
    sidecar={"agent_backend":backend,"session_id":sid,"codex_pid":999001,"broker_pid":999002,"cwd":"/workspace","log_path":str(log),"start_ts":100.0,"owner":"terminal","source":"final-blocker-cert"}
    (SOCKS/f'{sid}.json').write_text(json.dumps(sidecar)+'\n', encoding='utf-8')
    threading.Thread(target=serve,args=(sock,state),daemon=True).start()
    return log
cc_log=register('cc-live','cc',[cc_user('cc-live','silent live split')],{"busy":False,"queue_len":0,"interrupted_idle":False,"token":None})
fresh_log=register('fresh-interrupt','codex',codex_interrupted_rows(),{"busy":False,"queue_len":0,"interrupted_idle":True,"token":None})
manifest={"cc_log":str(cc_log),"fresh_log":str(fresh_log),"app_dir":str(APP)}
Path('/artifacts/manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({"ready_sockets":True, **manifest}), flush=True)
env=dict(os.environ); env['CODEX_WEB_HOST']='0.0.0.0'; env['CODEX_WEB_PORT']=str(PORT)
proc=subprocess.Popen([sys.executable,'-m','codoxear.server'],cwd='/workspace',env=env,stdout=open('/artifacts/server.stdout.log','w'),stderr=open('/artifacts/server.stderr.log','w'))
try:
    while proc.poll() is None: time.sleep(0.5)
finally:
    sys.exit(proc.returncode or 0)
