#!/usr/bin/env python3
"""Stateful fake broker for the interrupt->resume idle-projection claim (claim 5).

Models the broker state-machine transition that the real codoxear broker makes:
  Phase 0 (post-interrupt idle): busy=False, interrupted_idle=True
    -> sidebar/API project idle-with-interrupted-idle.
  On /send (resumed activity): busy=True, interrupted_idle=False
    -> sidebar/API project RUNNING, not falsely idle.
  After turn completion: busy=False, interrupted_idle=False
    -> clean idle, not falsely busy.

The state is driven by the same control-socket protocol the server uses
({"cmd":"state"} polls, {"cmd":"send"} resumes), so the real server discovery
and /api/sessions projection observe genuine transitions.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

HOME = Path(os.environ.get("HOME", "/home/tester"))
APP = HOME / ".local/share/codoxear"
SOCKS = APP / "socks"
LOGDIR = HOME / "cert-logs"
SOCKS.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)
PID = os.getpid()
SID = "cert-interrupt"
LOG = LOGDIR / f"{SID}.jsonl"


def write_log(rows):
    LOG.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def append_log(row):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# Initial log: an interrupted turn (user message, no completion).
write_log([
    {"type": "event_msg", "ts": 1.0,
     "payload": {"type": "user_message", "message": "work that got interrupted"}},
])

sidecar = {
    "agent_backend": "codex", "session_id": SID, "broker_pid": PID, "codex_pid": PID,
    "cwd": "/workspace/interrupt", "log_path": str(LOG), "start_ts": time.time(),
    "owner": "terminal", "sock_path": str(SOCKS / f"{SID}.sock"),
    "control_protocol_version": 2,
    "control_capabilities": {"sync_send": True, "key_write_errors": True},
}
(SOCKS / f"{SID}.json").write_text(json.dumps(sidecar), encoding="utf-8")

# Shared mutable state.
lock = threading.Lock()
state = {"busy": False, "interrupted_idle": True, "phase": 0}


def send_json(conn, obj):
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def serve():
    sp = SOCKS / f"{SID}.sock"
    if sp.exists():
        sp.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sp))
    srv.listen(8)
    srv.settimeout(0.5)
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            line = conn.makefile("rb").readline()
            if not line:
                continue
            req = json.loads(line.decode())
            cmd = req.get("cmd")
            with lock:
                if cmd == "state":
                    send_json(conn, {"busy": state["busy"], "queue_len": 0,
                                     "token": None,
                                     "interrupted_idle": state["interrupted_idle"]})
                elif cmd == "send":
                    # Resumed activity: flip to busy, clear interrupted_idle.
                    # The log already has an open user_message (interrupted turn)
                    # so log_idle=False; with broker busy=True the resolver
                    # projects busy=True (NOT falsely idle).
                    state["busy"] = True
                    state["interrupted_idle"] = False
                    state["phase"] = 1
                    send_json(conn, {"queued": False, "queue_len": 0, "busy": True})
                    # Schedule turn completion after the server observes busy.
                    def complete():
                        time.sleep(4.0)
                        # Append a final assistant answer + close so the log
                        # becomes idle (log_idle=True) -> busy=False cleanly.
                        append_log({"type": "event_msg", "ts": time.time(),
                                    "payload": {"type": "agent_message", "phase": "final_answer",
                                                "message": "resumed and completed"}})
                        append_log({"type": "event_msg", "ts": time.time() + 0.05,
                                    "payload": {"type": "task_complete", "turn_id": "t2",
                                                "last_agent_message": "resumed and completed"}})
                        with lock:
                            state["busy"] = False
                            state["interrupted_idle"] = False
                            state["phase"] = 2
                    threading.Thread(target=complete, daemon=True).start()
                elif cmd == "keys":
                    send_json(conn, {"ok": True, "queued": False, "n": 0, "key_queue_len": 0})
                elif cmd == "shutdown":
                    send_json(conn, {"ok": True})
                else:
                    send_json(conn, {"error": "unknown cmd"})
        except Exception as e:
            sys.stderr.write(f"stateful broker error: {e}\n")
        finally:
            try:
                conn.close()
            except Exception:
                pass


threading.Thread(target=serve, daemon=True).start()
sys.stderr.write(f"STATEFUL_BROKER_READY sid={SID} phase0(interrupted_idle=True,busy=False)\n")
sys.stderr.flush()
while True:
    time.sleep(1)
