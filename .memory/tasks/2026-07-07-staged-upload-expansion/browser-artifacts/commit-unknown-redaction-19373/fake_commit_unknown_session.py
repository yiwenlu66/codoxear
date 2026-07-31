#!/usr/bin/env python3
from __future__ import annotations
import json, os, socket, sys, threading, time
from pathlib import Path
HOME = Path(os.environ.get('HOME','/home/tester'))
APP = HOME/'.local/share/codoxear'
SOCKS = APP/'socks'
LOGDIR = HOME/'commit-unknown-proof-logs'
CWD = HOME/'commit-unknown-proof-repo'
CALLS = HOME/'commit-unknown-proof-calls.jsonl'
for p in (SOCKS, LOGDIR, CWD): p.mkdir(parents=True, exist_ok=True)
(CWD/'README.md').write_text('commit unknown redaction proof repo\n', encoding='utf-8')
SID='commit-unknown-redaction'
log_path=LOGDIR/f'{SID}.jsonl'
rows=[
 {'type':'event_msg','ts':1.0,'payload':{'type':'user_message','message':'commit unknown fixture'}},
 {'type':'response_item','ts':2.0,'payload':{'type':'message','role':'assistant','phase':'final_answer','content':[{'type':'output_text','text':'ready'}]}},
 {'type':'event_msg','ts':3.0,'payload':{'type':'task_complete','turn_id':'t1'}},
]
log_path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
sidecar={
 'agent_backend':'codex','session_id':SID,'broker_pid':os.getpid(),'codex_pid':os.getpid(),'cwd':str(CWD),
 'log_path':str(log_path),'start_ts':time.time(),'owner':'terminal','sock_path':str(SOCKS/f'{SID}.sock'),
 'control_protocol_version':2,
 'control_capabilities':{'sync_send':True,'key_write_errors':False},
}
(SOCKS/f'{SID}.json').write_text(json.dumps(sidecar), encoding='utf-8')
state={'busy':False,'queue_len':0,'token':None,'interrupted_idle':False}
sock_path=SOCKS/f'{SID}.sock'
if sock_path.exists(): sock_path.unlink()
srv=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(str(sock_path)); srv.listen(8); srv.settimeout(.5)
def log(req, resp):
    with CALLS.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'ts':time.time(),'req':req,'resp':resp})+'\n')
def send(conn,obj): conn.sendall((json.dumps(obj)+'\n').encode())
def loop():
  while True:
    try: conn,_=srv.accept()
    except socket.timeout: continue
    except OSError: break
    try:
      line=conn.makefile('rb').readline()
      req=json.loads(line.decode() or '{}') if line else {}
      cmd=req.get('cmd')
      if cmd=='state': resp=dict(state)
      elif cmd=='tail': resp={'tail':''}
      elif cmd=='send': resp={'commit_unknown':True,'queue_len':0}
      elif cmd=='keys': resp={'ok':True,'queued':False,'n':len(str(req.get('seq') or '')),'key_queue_len':0}
      elif cmd=='shutdown': resp={'ok':True}
      else: resp={'error':'unknown cmd'}
      log(req, resp); send(conn, resp)
    except Exception as e:
      print('fake broker error', e, file=sys.stderr)
    finally:
      try: conn.close()
      except Exception: pass
threading.Thread(target=loop, daemon=True).start()
print(json.dumps({'sid':SID,'cwd':str(CWD),'calls':str(CALLS),'sock':str(sock_path)}), flush=True)
while True: time.sleep(1)
