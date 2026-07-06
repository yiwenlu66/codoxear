#!/usr/bin/env python3
from __future__ import annotations
import json, os, socket, subprocess, sys, threading, time
from pathlib import Path
HOME = Path(os.environ.get('HOME','/home/tester'))
APP = HOME/'.local/share/codoxear'
SOCKS = APP/'socks'
LOGDIR = HOME/'monaco-proof-logs'
REPO = HOME/'monaco-proof-repo'
SOCKS.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)
REPO.mkdir(parents=True, exist_ok=True)
# Construct a real git repo for repository diff routes.
subprocess.run(['git','init'], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
subprocess.run(['git','config','user.email','proof@example.invalid'], cwd=REPO, check=True)
subprocess.run(['git','config','user.name','Proof'], cwd=REPO, check=True)
(REPO/'notes.txt').write_text('alpha line\nbeta line\ngamma line\n', encoding='utf-8')
(REPO/'changed.md').write_text('# Title\nbaseline body one\nbaseline body two\n', encoding='utf-8')
subprocess.run(['git','add','notes.txt','changed.md'], cwd=REPO, check=True)
subprocess.run(['git','commit','-m','baseline'], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
(REPO/'changed.md').write_text('# Title\nbaseline body one\nbaseline body two changed\nNEW monaco diff line\n', encoding='utf-8')
SID = 'monaco-proof'
log_path = LOGDIR/f'{SID}.jsonl'
rows = [
  {'type':'event_msg','ts':1.0,'payload':{'type':'user_message','message':'monaco proof fixture'}},
  {'type':'response_item','ts':2.0,'payload':{'type':'message','role':'assistant','phase':'final_answer','content':[{'type':'output_text','text':'fixture ready'}]}},
  {'type':'event_msg','ts':3.0,'payload':{'type':'task_complete','turn_id':'t1'}},
]
log_path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
sidecar = {
 'agent_backend':'codex',
 'session_id':SID,
 'broker_pid':os.getpid(),
 'codex_pid':os.getpid(),
 'cwd':str(REPO),
 'log_path':str(log_path),
 'start_ts':time.time(),
 'owner':'terminal',
 'sock_path':str(SOCKS/f'{SID}.sock'),
 'control_protocol_version':2,
 'control_capabilities':{'sync_send':True,'key_write_errors':True},
}
(SOCKS/f'{SID}.json').write_text(json.dumps(sidecar), encoding='utf-8')
state={'busy':False,'queue_len':0,'token':None,'interrupted_idle':False}
sock_path=SOCKS/f'{SID}.sock'
if sock_path.exists(): sock_path.unlink()
srv=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(str(sock_path)); srv.listen(8); srv.settimeout(.5)
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
      if cmd=='state': send(conn, state)
      elif cmd=='tail': send(conn, {'tail':''})
      elif cmd=='send': send(conn, {'queued':False,'queue_len':0})
      elif cmd=='keys': send(conn, {'ok':True,'queued':False,'n':0,'key_queue_len':0})
      elif cmd=='shutdown': send(conn, {'ok':True})
      else: send(conn, {'error':'unknown cmd'})
    except Exception as e:
      print('fake broker error', e, file=sys.stderr)
    finally:
      try: conn.close()
      except Exception: pass
threading.Thread(target=loop, daemon=True).start()
print(json.dumps({'sid':SID,'repo':str(REPO),'log_path':str(log_path),'sock_path':str(sock_path)}), flush=True)
while True: time.sleep(1)
