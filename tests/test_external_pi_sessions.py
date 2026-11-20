from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from codoxear.external_sessions import ExternalSessionRegistry, external_dir, process_matches
from codoxear import pi_extension
from test_pi_bridge_session_name import _bundle, _request


def wait_for(fn):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        if value := fn():
            return value
        time.sleep(0.02)
    raise AssertionError("bridge did not become ready")


@pytest.fixture
def external_bridge(tmp_path):
    bundle = tmp_path / "bridge.cjs"
    _bundle(bundle)
    harness = tmp_path / "harness.cjs"
    harness.write_text('''
const fs = require('node:fs');
const bridge = require(process.argv[2]).default;
const folder = process.argv[3];
const handlers = new Map();
let registrations = 0, busy = false, name = '', sends = [];
const sessionFile = folder + '/session.jsonl';
const header = {type:'session', id:'one', cwd:folder};
const settings = {type:'model_change', modelId:'test', provider:'local'};
const ctx = {
  sessionManager: {getSessionFile:()=>sessionFile, getSessionId:()=> 'one', getCwd:()=>folder, fileEntries:[header,settings], flushed:false},
  isIdle:()=>!busy, abort:()=>{busy=false;}, ui:{notify(){}}, model:{id:'test',provider:'local'}
};
const pi = {
  on(event, fn) { registrations++; handlers.set(event,fn); },
  registerCommand(){}, getCommands:()=>[{name:'model'},{name:'custom-action'}], getThinkingLevel:()=> 'low',
  getSessionName:()=>name, setSessionName:(v)=>{name=v;}, setThinkingLevel(){},
  sendUserMessage(text, options) { sends.push({text,options}); busy=true; fs.writeFileSync(folder+'/sends.json',JSON.stringify(sends)); }
};
bridge(pi); const count=registrations; bridge(pi);
if(count!==registrations) throw new Error('duplicate listeners');
handlers.get('session_start')({reason:'startup'},ctx);
process.stdin.resume();
process.stdin.on('data',()=>{handlers.get('session_shutdown')({},ctx); process.exit(0);});
''')
    env = {**os.environ, "CODOXEAR_EXTERNAL_DIR": str(tmp_path)}
    for key in ("CODEX_WEB_PI_ACTIVE_SESSION_FILE", "CODEX_WEB_PI_BRIDGE_PID", "PI_SUBAGENT_CHILD"):
        env.pop(key, None)
    process = subprocess.Popen(["node", str(harness), str(bundle), str(tmp_path)], stdin=subprocess.PIPE, env=env)
    socket = tmp_path / f"pi-{process.pid}.bridge.sock"
    marker = tmp_path / f"pi-{process.pid}.json"
    wait_for(lambda: socket.exists() and marker.exists())
    yield process, socket, marker
    if process.poll() is None:
        process.communicate(b"shutdown", timeout=5)


def test_external_bridge_lifecycle_control_and_registry(external_bridge, tmp_path):
    process, socket, marker = external_bridge
    state = _request(socket, {"op": "state"})
    assert state["busy"] is False
    assert state["sessionId"] == "one"
    assert _request(socket, {"op": "send", "text": "normal user"})["delivery"] == "turn"
    assert _request(socket, {"op": "state"})["busy"] is True
    assert _request(socket, {"op": "send", "text": "steer user"})["delivery"] == "steer"
    sends = json.loads((tmp_path / "sends.json").read_text())
    assert sends == [{"text": "normal user", "options": {"expandPromptTemplates": True}},
                     {"text": "steer user", "options": {"deliverAs": "steer", "expandPromptTemplates": True}}]
    assert _request(socket, {"op": "send", "text": "stale", "sessionId": "old"})["error"] == "stale session"
    assert _request(socket, {"op": "interrupt"})["ok"]
    assert not _request(socket, {"op": "state"})["busy"]
    assert _request(socket, {"op": "set_session_name", "name": "External title"})["name"] == "External title"
    # Startup materializes the complete buffer, not just the header.
    rows = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text().splitlines()]
    assert [row["type"] for row in rows] == ["session", "model_change"]
    registry = ExternalSessionRegistry(tmp_path)
    row, = registry.rows()
    assert row["external"] and row["display_name"] == "External title"
    assert row["slash_commands"] == [{"name": "custom-action", "description": ""}]
    sid = row["session_id"]
    assert registry.get(sid).sock_path == socket
    assert registry.dismiss(sid)
    assert process.poll() is None
    assert registry.rows() == []
    process.communicate(b"shutdown", timeout=5)
    assert not marker.exists() and not socket.exists()


def test_external_commands_are_offered_only_when_pi_can_dispatch_them(external_bridge, tmp_path):
    _, socket, _ = external_bridge
    registry = ExternalSessionRegistry(tmp_path)
    assert registry.rows()[0]["slash_commands"]
    _request(socket, {"op": "send", "text": "start run"})
    registry.scan(force=True)
    assert registry.rows()[0]["slash_commands"] == []
    _request(socket, {"op": "interrupt"})
    registry.scan(force=True)
    assert registry.rows()[0]["slash_commands"] == [{"name": "custom-action", "description": ""}]


def test_stale_pid_incarnation_is_not_a_session(external_bridge, tmp_path):
    _, _, marker = external_bridge
    data = json.loads(marker.read_text())
    data["processStart"] = "wrong-incarnation"
    marker.write_text(json.dumps(data))
    assert ExternalSessionRegistry(tmp_path).rows() == []
    assert not process_matches(99999999, None)


def test_installer_live_loader_override_and_dangling_target(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "custom-pi"))
    assert pi_extension.extension_status()["status"] == "missing"
    status = pi_extension.install()
    assert status["status"] == "ok"
    assert Path(status["loader"]) == tmp_path / "custom-pi/extensions/codoxear-bridge.ts"
    assert Path(status["target"]) == pi_extension.bridge_path()
    Path(status["loader"]).write_text(pi_extension.loader_content(tmp_path / "gone.ts"))
    assert pi_extension.extension_status()["status"] == "error"
    assert pi_extension.install()["status"] == "ok"
    assert pi_extension.uninstall()["status"] == "missing"
    Path(status["loader"]).write_text("// user's own file")
    with pytest.raises(ValueError, match="unmanaged"):
        pi_extension.install()


def test_doctor_default_on_install_and_explicit_opt_out(tmp_path, monkeypatch, capfd):
    from codoxear import dependency_doctor as doctor
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    assert doctor.main(["--install", "--git", "--yes", "--json"]) == 0
    report = json.loads(capfd.readouterr().out)
    assert report["checks"]["pi-extension"]["status"] == "ok"
    assert report["plan"]["groups"] == ["pi-extension"]
    pi_extension.uninstall()
    assert doctor.main(["--dry-run", "--git", "--no-pi-extension", "--json"]) == 0
    assert "pi-extension" not in json.loads(capfd.readouterr().out)["checks"]
    assert pi_extension.extension_status()["status"] == "missing"


def test_xdg_external_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("CODOXEAR_EXTERNAL_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert external_dir() == tmp_path / "codoxear/external"
