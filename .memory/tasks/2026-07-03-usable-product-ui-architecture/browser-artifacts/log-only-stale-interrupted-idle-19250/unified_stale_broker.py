#!/usr/bin/env python3
"""Unified stale fake broker for the log-only interrupted_idle verification.

Control file /tmp/stale_broker_ctrl selects the reported interrupted_idle value:
  - missing/empty/"true"  -> interrupted_idle=true  (stale, phases 1-2)
  - "false"               -> interrupted_idle=false (phase 3a: broker clears)

busy is ALWAYS false, queue_len ALWAYS 0. Writes socket + sidecar + initial
interrupted-turn log under the container app dir so the REAL server discovers
them. No product code is stubbed.
"""
from __future__ import annotations
import json, os, socket, sys, threading, time
from pathlib import Path

HOME = Path("/home/tester")
APP = HOME / ".local/share/codoxear"
SOCKS = APP / "socks"
LOGDIR = HOME / "cert-logs"
SOCKS.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

PID = os.getpid()
SID = "cert-stale-interrupt"
LOG = LOGDIR / f"{SID}.jsonl"
SOCK = SOCKS / f"{SID}.sock"
SIDECAR = SOCKS / f"{SID}.json"
CTRL = Path("/tmp/stale_broker_ctrl")


def write_log(rows):
    LOG.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# Fresh interrupted turn: user + non-final assistant fragment (no completion).
write_log([
    {"type": "session_meta", "payload": {"id": SID, "source": "cli"}},
    {"type": "event_msg", "ts": 10.0,
     "payload": {"type": "user_message", "message": "work that got interrupted"}},
    {"type": "response_item", "ts": 11.0, "payload": {
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "thinking about"}]}},
])

sidecar = {
    "agent_backend": "codex", "session_id": SID, "thread_id": SID,
    "broker_pid": PID, "codex_pid": PID, "cwd": "/home/tester",
    "log_path": str(LOG), "start_ts": time.time(), "owner": "terminal",
    "sock_path": str(SOCK), "control_protocol_version": 2,
    "control_capabilities": {"sync_send": True, "key_write_errors": True},
}
SIDECAR.write_text(json.dumps(sidecar), encoding="utf-8")


def ii_value():
    try:
        return CTRL.read_text().strip() != "false"
    except Exception:
        return True


def send_json(conn, obj):
    conn.sendall((json.dumps(obj) + "\n").encode())


def serve():
    if SOCK.exists():
        SOCK.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK)); srv.listen(8); srv.settimeout(0.5)
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
            req = json.loads(line.decode()); cmd = req.get("cmd")
            if cmd == "state":
                send_json(conn, {"busy": False, "queue_len": 0, "token": None,
                                 "interrupted_idle": ii_value()})
            elif cmd == "tail":
                send_json(conn, {"tail": ""})
            elif cmd == "send":
                send_json(conn, {"queued": False, "queue_len": 0})
            elif cmd == "keys":
                send_json(conn, {"ok": True, "queued": False, "n": 0, "key_queue_len": 0})
            elif cmd == "shutdown":
                send_json(conn, {"ok": True})
            else:
                send_json(conn, {"error": "unknown cmd"})
        except Exception as e:
            sys.stderr.write(f"broker error: {e}\n"); sys.stderr.flush()
        finally:
            try:
                conn.close()
            except Exception:
                pass


threading.Thread(target=serve, daemon=True).start()
Path("/tmp/stale_broker_ready").write_text(
    f"READY sid={SID} sock={SOCK} log={LOG} pid={PID}\n", encoding="utf-8")
sys.stderr.write(f"STALE_BROKER_READY sid={SID} sock={SOCK} log={LOG} pid={PID}\n")
sys.stderr.flush()
while True:
    time.sleep(1)
