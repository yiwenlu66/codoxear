from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from codoxear.external_sessions import ExternalSessionRegistry
from test_pi_bridge_session_name import _bundle, _request


@pytest.fixture(params=[False, True], ids=["sdk-lazy", "broker-shared-bus"])
def host(request, tmp_path):
    bundle = tmp_path / "bridge.cjs"
    _bundle(bundle)
    harness = tmp_path / "host.cjs"
    harness.write_text('''
const fs = require('node:fs');
const bridge = require(process.argv[2]);
const root = process.argv[3], broker = process.argv[4] === 'true';
const busHandlers = new Map();
const events = {
  on(name, fn) { const set=busHandlers.get(name)||new Set(); set.add(fn); busHandlers.set(name,set); return ()=>set.delete(fn); },
  emit(name, value) { for(const fn of [...(busHandlers.get(name)||[])]) fn(value); }
};
if(broker) {
  process.env.CODEX_WEB_PI_ACTIVE_SESSION_FILE = root+'/broker.json';
  process.env.CODEX_WEB_PI_BRIDGE_PID = String(process.pid);
}
function runtime(key) {
  const handlers=new Map();
  const state={id:key,busy:false,disposed:false,name:key,commands:0};
  const manager={
    getSessionId:()=>state.id, getSessionFile:()=>root+'/'+state.id+'.jsonl', getCwd:()=>root+'/'+key,
    fileEntries:[{type:'session',id:key,cwd:root+'/'+key}], flushed:false
  };
  function alive(){if(state.disposed)throw Error('disposed ctx');}
  const ctx={
    get sessionManager(){alive();return manager;}, isIdle(){alive();return !state.busy;},
    abort(){alive();state.busy=false;}, model:{provider:'local',id:key},ui:{notify(){}}
  };
  const api=()=>({
    on(name,fn){const list=handlers.get(name)||[];list.push(fn);handlers.set(name,list);}, events,
    registerCommand(){state.commands++;}, getCommands:()=>[], getThinkingLevel:()=>key==='a'?'low':'high',setThinkingLevel(){},
    getSessionName:()=>state.name,setSessionName:(name)=>{state.name=name;},
    sendUserMessage(text,options){alive();state.busy=true;fs.appendFileSync(root+'/'+key+'-sends.jsonl',JSON.stringify({text,options})+'\\n');}
  });
  return {state,manager,ctx,api,fire:(name,event={})=>{for(const fn of handlers.get(name)||[])fn(event,ctx);}};
}
let a=runtime('a'); const b=runtime('b');
bridge.globalBridge(a.api());
if(broker) bridge.default(a.api());
bridge.globalBridge(b.api());
// Do not emit session_start. Each fallback must independently create runtime.
a.fire('turn_start'); b.fire('agent_start');
const initialFiles=fs.readdirSync(root).filter(n=>/^pi-.*\\.json$/.test(n)).map(n=>root+'/'+n);
console.log(JSON.stringify({pid:process.pid,files:initialFiles,broker:broker?root+'/broker.json':null,commands:a.state.commands}));
process.stdin.setEncoding('utf8');
process.stdin.on('data',(command)=>{
  if(command.trim()==='rebind') {
    a.state.id='a-new';a.manager.fileEntries=[{type:'session',id:'a-new',cwd:root+'/a'}];
    a.fire('turn_start'); console.log('rebound');
  } else if(command.trim()==='replace-a') {
    a.fire('session_shutdown',{reason:'new',targetSessionFile:root+'/a-next.jsonl'});
    const remaining=fs.readdirSync(root).filter(n=>/^pi-.*\\.json$/.test(n));
    if(remaining.some(n=>JSON.parse(fs.readFileSync(root+'/'+n)).sessionId==='a-new')) throw Error('old marker still published');
    // Another SDK instance registering first must not steal the pending row.
    const unrelated=runtime('unrelated');bridge.globalBridge(unrelated.api());unrelated.fire('turn_start');
    unrelated.fire('session_shutdown',{reason:'quit'});
    a=runtime('a-next');bridge.globalBridge(a.api());if(broker)bridge.default(a.api());
    a.fire('agent_start');console.log('replaced');
  } else if(command.trim()==='reload-a') {
    a.fire('session_shutdown',{reason:'reload'});
    a=runtime('a-next');bridge.globalBridge(a.api());if(broker)bridge.default(a.api());
    a.fire('turn_start');console.log('reloaded');
  } else if(command.trim()==='expire-a' || command.trim()==='no-target-a') {
    const expire=command.trim()==='expire-a',key=expire?'a-expired':'a-unknown';
    a.fire('session_shutdown',{reason:'new',...(expire?{targetSessionFile:root+'/'+key+'.jsonl'}:{})});
    const realNow=Date.now;if(expire)Date.now=()=>realNow()+11000;
    a=runtime(key);bridge.globalBridge(a.api());if(broker)bridge.default(a.api());a.fire('turn_start');
    Date.now=realNow;console.log(key);
  } else if(command.trim()==='dispose-a') {a.state.disposed=true;console.log('disposed');}
  else if(command.trim()==='shutdown-b') {b.fire('session_shutdown');console.log('shutdown');}
  else process.exit(0);
});
''')
    env = {**os.environ, "CODOXEAR_EXTERNAL_DIR": str(tmp_path)}
    for key in ("CODEX_WEB_PI_ACTIVE_SESSION_FILE", "CODEX_WEB_PI_BRIDGE_PID", "PI_SUBAGENT_CHILD"):
        env.pop(key, None)
    process = subprocess.Popen(["node", str(harness), str(bundle), str(tmp_path), str(request.param).lower()],
                               env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    info = json.loads(process.stdout.readline())
    yield process, info, tmp_path
    process.communicate("exit\n", timeout=5)


def test_instance_isolation_laziness_rebind_and_disposal(host):
    process, info, root = host
    registry = ExternalSessionRegistry(root)
    rows = registry.rows()
    broker = info["broker"]
    assert len(rows) == (1 if broker else 2)
    b = next(row for row in rows if row["thread_id"] == "b")
    a = next((row for row in rows if row["thread_id"] == "a"), None)
    a_socket = Path(f"{broker}.bridge.sock") if broker else Path(info["files"][0]).with_suffix(".bridge.sock")
    if not broker:
        a_socket = registry.require(a["session_id"]).sock_path
        assert a["session_id"] != b["session_id"]
        assert a["external_instance_id"] != b["external_instance_id"]
    else:
        assert info["commands"] == 2, "global twin registered broker commands before yielding ownership"
    b_socket = registry.require(b["session_id"]).sock_path
    assert _request(a_socket, {"op": "state"})["model"] == "a"
    assert _request(b_socket, {"op": "state"})["model"] == "b"
    _request(a_socket, {"op": "send", "text": "only-a"})
    _request(b_socket, {"op": "send", "text": "only-b"})
    assert "only-a" in (root / "a-sends.jsonl").read_text()
    assert "only-a" not in (root / "b-sends.jsonl").read_text()
    _request(a_socket, {"op": "interrupt"})
    assert not _request(a_socket, {"op": "state"})["busy"]
    assert _request(b_socket, {"op": "state"})["busy"]
    process.stdin.write("rebind\n")
    process.stdin.flush()
    assert process.stdout.readline().strip() == "rebound"
    registry.scan(force=True)
    assert _request(a_socket, {"op": "state"})["sessionId"] == "a-new"
    if not broker:
        assert registry.require(a["session_id"]).thread_id == "a-new"
    assert registry.require(b["session_id"]).thread_id == "b"
    assert _request(a_socket, {"op": "send", "text": "stale", "sessionId": "a"})["error"] == "stale session"
    for command, expected in (("replace-a", "replaced"), ("reload-a", "reloaded")):
        process.stdin.write(command + "\n")
        process.stdin.flush()
        assert process.stdout.readline().strip() == expected
        registry.scan(force=True)
        assert _request(a_socket, {"op": "state"})["sessionId"] == "a-next"
        if not broker:
            assert registry.require(a["session_id"]).thread_id == "a-next"
        assert registry.require(b["session_id"]).thread_id == "b"
    process.stdin.write("dispose-a\n")
    process.stdin.flush()
    assert process.stdout.readline().strip() == "disposed"
    if broker:
        assert _request(a_socket, {"op": "state"})["ok"] is False
    registry.scan(force=True)
    assert [row["thread_id"] for row in registry.rows()] == ["b"]
    assert process.poll() is None
    process.stdin.write("shutdown-b\n")
    process.stdin.flush()
    assert process.stdout.readline().strip() == "shutdown"
    registry.scan(force=True)
    assert registry.rows() == []


def test_handoff_never_adopts_expired_or_unspecified_destinations(host):
    process, info, root = host
    registry = ExternalSessionRegistry(root)
    broker = info["broker"]
    rows = registry.rows()
    a = next((row for row in rows if row["thread_id"] == "a"), None)
    socket = Path(f"{broker}.bridge.sock") if broker else registry.require(a["session_id"]).sock_path
    identity = _request(socket, {"op": "state"})["instanceId"]
    b_id = next(row["session_id"] for row in rows if row["thread_id"] == "b")
    for command, target in (("expire-a", "a-expired"), ("no-target-a", "a-unknown")):
        old_socket = socket
        process.stdin.write(command + "\n")
        process.stdin.flush()
        assert process.stdout.readline().strip() == target
        registry.scan(force=True)
        if not broker:
            assert not old_socket.exists()
            new_row = next(row for row in registry.rows() if row["thread_id"] == target)
            assert new_row["session_id"] != a["session_id"]
            socket = registry.require(new_row["session_id"]).sock_path
        next_identity = _request(socket, {"op": "state"})["instanceId"]
        assert next_identity != identity
        identity = next_identity
        assert registry.require(b_id).thread_id == "b"
