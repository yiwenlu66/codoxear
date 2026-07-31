#!/usr/bin/env python3
from __future__ import annotations
import json, os, socket, sys, threading, time
from pathlib import Path
HOME = Path(os.environ.get('HOME','/home/tester'))
APP = HOME/'.local/share/codoxear'
SOCKS = APP/'socks'
CLAUDE = HOME/'.claude'
PROJECTS = CLAUDE/'projects'
CWD = HOME/'cc-context-proof-repo'
CALLS = HOME/'cc-context-proof-calls.jsonl'
SID='cc-context-proof'
THREAD='11111111-2222-4333-8444-555555555555'
for p in (SOCKS, PROJECTS, CWD): p.mkdir(parents=True, exist_ok=True)
(CWD/'README.md').write_text('cc context proof repo\n', encoding='utf-8')
log_dir = PROJECTS/'-home-tester-cc-context-proof-repo'
log_dir.mkdir(parents=True, exist_ok=True)
log_path=log_dir/f'{THREAD}.jsonl'
rows=[
 {'type':'user','sessionId':THREAD,'timestamp':'2026-07-07T04:55:00.000Z','cwd':str(CWD),'message':{'role':'user','content':'cc context proof fixture'}},
 {'type':'assistant','sessionId':THREAD,'timestamp':'2026-07-07T04:55:10.000Z','message':{'role':'assistant','model':'claude-sonnet-4-5','content':[{'type':'text','text':'ready'}],'stop_reason':'end_turn','usage':{'input_tokens':100000,'cache_read_input_tokens':20000,'cache_creation_input_tokens':30000,'output_tokens':9999,'service_tier':'standard'}}},
 {'type':'system','subtype':'turn_duration','sessionId':THREAD,'timestamp':'2026-07-07T04:55:11.000Z','durationMs':1000},
]
log_path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
sidecar={
 'agent_backend':'cc','session_id':SID,'thread_id':THREAD,'broker_pid':os.getpid(),'codex_pid':os.getpid(),'claude_pid':os.getpid(),'cwd':str(CWD),
 'log_path':str(log_path),'start_ts':time.time(),'owner':'terminal','sock_path':str(SOCKS/f'{SID}.sock'),
 'control_protocol_version':2,'control_capabilities':{'sync_send':True,'key_write_errors':False},
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
      elif cmd=='send': resp={'queued':False,'queue_len':0,'busy':True}
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
print(json.dumps({'sid':SID,'thread':THREAD,'cwd':str(CWD),'log':str(log_path),'calls':str(CALLS),'sock':str(sock_path)}), flush=True)
while True: time.sleep(1)
