"""Execute the Pi bridge and broker receipt path; no source-text assertions."""

import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from codoxear import broker as broker_mod
from codoxear import sessiond as sessiond_mod
from codoxear.broker import Broker
from codoxear.broker_launch import _read_pi_prompt_receipt, _reset_pi_active_session_marker
from codoxear.broker_turn_state import State, arm_send_latch
from codoxear.sessiond_state import State as SessiondState
from codoxear.sessiond import Sessiond


ROOT = Path(__file__).resolve().parents[1]
TEXT = "第一条 prompt\nline two"
DIGEST = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()


def _payload(**changes):
    return {"version": 1, "pid": 42, "textSha256": DIGEST, "messageTimestamp": 10001, **changes}


def _write(marker, payload):
    Path(f"{marker}.receipt").write_text(json.dumps(payload), encoding="utf-8")


def test_bridge_writes_user_ingestion_receipt_without_materializing_rollout(tmp_path):
    bundle = tmp_path / "bridge.cjs"
    subprocess.run([
        "npx", "esbuild", str(ROOT / "codoxear/pi_active_session_bridge.ts"),
        "--bundle", "--platform=node", "--format=cjs", f"--outfile={bundle}", "--log-level=error",
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    marker = tmp_path / "active.json"
    script = r'''
const fs = require("node:fs");
const bridge = require(process.argv[1]).default;
const handlers = new Map();
bridge({
  on(name, handler) { handlers.set(name, handler); },
  registerCommand() {}, getCommands() { return []; }, getThinkingLevel() { return "off"; },
});
const receipt = `${process.env.CODEX_WEB_PI_ACTIVE_SESSION_FILE}.receipt`;
const read = () => JSON.parse(fs.readFileSync(receipt, "utf8"));
const start = (message) => handlers.get("message_start")({type: "message_start", message});
start({role: "assistant", content: "answer", timestamp: 10001});
const ignoredAssistant = !fs.existsSync(receipt);
start({role: "user", content: "invalid timestamp", timestamp: NaN});
const ignoredInvalid = !fs.existsSync(receipt);
start({role: "user", content: [{type: "text", text: "第一条 prompt\n"}, {type: "image", data: "ignored"}, {type: "text", text: "line two"}], timestamp: 10001});
const first = read();
const ctx = { sessionManager: {
  getSessionFile() { return process.env.SESSION_FILE; },
  getSessionId() { return "session"; }, getCwd() { return "/tmp"; },
}, ui: {notify() {}} };
handlers.get("session_start")({type: "session_start", reason: "startup"}, ctx);
handlers.get("turn_end")({type: "turn_end", turnIndex: 0}, ctx);
const retained = read();
const mode = fs.statSync(receipt).mode & 0o777;
start({role: "toolResult", content: "result", timestamp: 20001});
const afterTool = read();
console.log(JSON.stringify({ignoredAssistant, ignoredInvalid, first, retained, afterTool, mode, pid: process.pid, rolloutExists: fs.existsSync(process.env.SESSION_FILE)}));
'''
    result = subprocess.run(["node", "-e", script, str(bundle)], check=True, capture_output=True, text=True,
                            env={**os.environ, "CODEX_WEB_PI_ACTIVE_SESSION_FILE": str(marker), "SESSION_FILE": str(tmp_path / "absent.jsonl")})
    observed = json.loads(result.stdout)
    assert observed["ignoredAssistant"] and observed["ignoredInvalid"]
    assert observed["first"] == {"version": 1, "pid": observed["pid"], "textSha256": DIGEST, "messageTimestamp": 10001}
    assert observed["retained"] == observed["afterTool"] == observed["first"]
    assert observed["mode"] == 0o600
    assert not observed["rolloutExists"]
    assert _read_pi_prompt_receipt(marker, process_pid=observed["pid"]) == (DIGEST, 10001)
    # The retained first receipt survives session/caps rewrites and is consumed
    # by the real broker sweep even though the rollout is still absent.
    state = _armed_state(tmp_path)
    state.codex_pid = observed["pid"]
    broker = Broker.__new__(Broker)
    broker.state = state
    broker._lock = threading.Lock()
    broker.pi_active_session_marker_path = marker
    with patch.object(broker_mod, "AGENT_BACKEND", "pi"), patch.object(broker_mod, "_now", return_value=14.0):
        broker._mark_idle_if_quiet()
    assert state.send_latch_accepted is True
    assert state.send_verdict is None


@pytest.mark.parametrize("changes", [
    {"version": 2}, {"version": True}, {"pid": 7}, {"pid": True},
    {"textSha256": "bad"}, {"textSha256": None}, {"messageTimestamp": True},
    {"messageTimestamp": 0}, {"messageTimestamp": 10.5}, {"messageTimestamp": 2**53},
])
def test_reader_rejects_foreign_or_malformed_receipts(tmp_path, changes):
    marker = tmp_path / "active.json"
    _write(marker, _payload(**changes))
    assert _read_pi_prompt_receipt(marker, process_pid=42) is None


def test_reader_absent_invalid_and_reset(tmp_path):
    marker = tmp_path / "active.json"
    assert _read_pi_prompt_receipt(marker, process_pid=42) is None
    Path(f"{marker}.receipt").write_text("{", encoding="utf-8")
    assert _read_pi_prompt_receipt(marker, process_pid=42) is None
    _write(marker, _payload())
    assert _read_pi_prompt_receipt(marker, process_pid=42) == (DIGEST, 10001)
    assert _read_pi_prompt_receipt(marker, process_pid=0) is None
    _reset_pi_active_session_marker(marker)
    assert _read_pi_prompt_receipt(marker, process_pid=42) is None


def _armed_state(root):
    state = State(codex_pid=42, pty_master_fd=1, cwd=str(root), start_ts=0.0,
                  codex_home=root, sessions_dir=root)
    arm_send_latch(state, send_id="first", text=TEXT, prev_busy=False, prev_turn_open=False)
    state.busy = state.turn_open = True
    state.send_latch_activity_ts = state.last_turn_activity_ts = 10.0
    return state


@pytest.mark.parametrize("pid,accepted", [(42, True), (7, False)])
def test_broker_reads_receipt_before_unbound_log_quiet_verdict(tmp_path, pid, accepted):
    marker = tmp_path / "active.json"
    _write(marker, _payload(pid=pid))
    state = _armed_state(tmp_path)
    broker = Broker.__new__(Broker)
    broker.state = state
    broker._lock = threading.Lock()
    broker.pi_active_session_marker_path = marker
    with patch.object(broker_mod, "AGENT_BACKEND", "pi"), patch.object(broker_mod, "_now", return_value=14.0):
        broker._mark_idle_if_quiet()
    assert state.send_latch_accepted is accepted
    # t=14.0 is 4s after the send: past the normal quiet window but inside the
    # extended first-send window (this session has never shown backend turn
    # activity), so even an unmatched receipt cannot produce a verdict yet —
    # cold first-prompt preflight legitimately hides every evidence channel.
    assert state.busy is True
    assert state.send_verdict is None


def test_broker_cold_first_send_verdict_waits_for_extended_window(tmp_path):
    marker = tmp_path / "active.json"
    state = _armed_state(tmp_path)
    broker = Broker.__new__(Broker)
    broker.state = state
    broker._lock = threading.Lock()
    broker.pi_active_session_marker_path = marker
    first_quiet = broker_mod.FIRST_SEND_QUIET_SECONDS
    with patch.object(broker_mod, "AGENT_BACKEND", "pi"), patch.object(
        broker_mod, "_now", return_value=10.0 + first_quiet - 0.01
    ):
        broker._mark_idle_if_quiet()
    assert state.busy is True
    assert state.send_verdict is None
    settle_ts = 10.0 + first_quiet + 0.01
    with patch.object(broker_mod, "AGENT_BACKEND", "pi"), patch.object(broker_mod, "_now", return_value=settle_ts):
        broker._mark_idle_if_quiet()
    assert state.busy is False
    assert state.send_verdict == {"send_id": "first", "verdict": "undelivered", "ts": settle_ts}


def test_broker_warm_session_send_settles_on_normal_window(tmp_path):
    marker = tmp_path / "active.json"
    state = _armed_state(tmp_path)
    # The backend already produced turn activity in this process, so cold
    # preflight cannot be the explanation for silence: the normal window applies.
    state.backend_turn_activity_seen = True
    broker = Broker.__new__(Broker)
    broker.state = state
    broker._lock = threading.Lock()
    broker.pi_active_session_marker_path = marker
    with patch.object(broker_mod, "AGENT_BACKEND", "pi"), patch.object(broker_mod, "_now", return_value=14.0):
        broker._mark_idle_if_quiet()
    assert state.busy is False
    assert state.send_verdict == {"send_id": "first", "verdict": "undelivered", "ts": 14.0}


def test_sessiond_reads_receipt_before_quiet_verdict(tmp_path):
    marker = tmp_path / "active.json"
    _write(marker, _payload())
    state = SessiondState(session_id=None, codex_pid=42, log_path=tmp_path / "not-persisted.jsonl",
                         sock_path=tmp_path / "broker.sock", pty_master_fd=1, start_ts=0.0)
    arm_send_latch(state, send_id="first", text=TEXT, prev_busy=False, prev_turn_open=False)
    state.busy = state.turn_open = True
    state.send_latch_activity_ts = state.last_turn_activity_ts = 10.0
    daemon = Sessiond.__new__(Sessiond)
    daemon.state = state
    daemon._lock = threading.Lock()
    daemon._stop = threading.Event()
    daemon.pi_active_session_marker_path = marker
    daemon._refresh_pi_thinking_capability = lambda: None
    with patch.object(sessiond_mod, "AGENT_BACKEND", "pi"), \
            patch.object(sessiond_mod, "_now", return_value=14.0), \
            patch.object(sessiond_mod, "_read_jsonl_from_offset", return_value=([], 0)), \
            patch.object(sessiond_mod.time, "sleep", side_effect=lambda _: daemon._stop.set()):
        daemon._log_watcher()
    assert state.send_latch_accepted is True
    assert state.busy is True
    assert state.send_verdict is None
