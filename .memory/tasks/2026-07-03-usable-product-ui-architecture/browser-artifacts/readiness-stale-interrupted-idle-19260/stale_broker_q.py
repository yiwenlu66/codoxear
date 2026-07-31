#!/usr/bin/env python3
"""Second stale broker instance for the clean queue discriminator.
Uses a distinct SID + fresh log so the prior send-boundary does not interfere.
Queue promotion is probed BEFORE any direct send.
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
SID = "cert-stale-q"
LOG = LOGDIR / f"{SID}.jsonl"
SOCK = SOCKS / f"{SID}.sock"
SIDECAR = SOCKS / f"{SID}.json"
CALLLOG = Path("/tmp/stale_broker_calls_q.jsonl")

LOG.write_text(
    "".join(json.dumps(r) + "\n" for r in [
        {"type": "session_meta", "payload": {"id": SID, "source": "cli"}},
        {"type": "event_msg", "ts": 10.0,
         "payload": {"type": "user_message", "message": "work that got interrupted"}},
        {"type": "response_item", "ts": 11.0, "payload": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "thinking about"}]}},
    ]),
    encoding="utf-8",
)

sidecar = {
    "agent_backend": "codex", "session_id": SID, "thread_id": SID,
    "broker_pid": PID, "codex_pid": PID, "cwd": "/home/tester",
    "log_path": str(LOG), "start_ts": time.time(), "owner": "terminal",
    "sock_path": str(SOCK), "control_protocol_version": 2,
    "control_capabilities": {"sync_send": True, "key_write_errors": True},
}
SIDECAR.write_text(json.dumps(sidecar), encoding="utf-8")
CALLLOG.write_text("", encoding="utf-8")
CALL_LOCK = threading.Lock()


def record_call(cmd, req, resp):
    e = {"ts": time.time(), "cmd": cmd, "req": req, "resp": resp}
    with CALL_LOCK:
        with CALLLOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e) + "\n")


def send_json(conn, obj):
    conn.sendall((json.dumps(obj) + "\n").encode())


def handle(conn):
    try:
        line = conn.makefile("rb").readline()
        if not line:
            return
        req = json.loads(line.decode())
        cmd = req.get("cmd")
        if cmd == "state":
            resp = {"busy": False, "queue_len": 0, "token": None,
                    "interrupted_idle": True}
            send_json(conn, resp); record_call(cmd, {"cmd": "state"}, resp)
        elif cmd == "tail":
            resp = {"tail": ""}; send_json(conn, resp); record_call(cmd, {}, resp)
        elif cmd == "send":
            resp = {"queued": False, "queue_len": 0, "busy": True}
            send_json(conn, resp)
            record_call(cmd, {"cmd": "send", "text": req.get("text"),
                              "queue_item_id": req.get("queue_item_id")}, resp)
        elif cmd == "keys":
            resp = {"ok": True, "queued": False, "n": 0, "key_queue_len": 0}
            send_json(conn, resp); record_call(cmd, {"cmd": "keys"}, resp)
        elif cmd == "shutdown":
            resp = {"ok": True}; send_json(conn, resp); record_call(cmd, {}, resp)
        else:
            resp = {"error": "unknown cmd"}; send_json(conn, resp); record_call(cmd, {}, resp)
    except Exception as e:
        sys.stderr.write(f"broker-q error: {e}\n"); sys.stderr.flush()
    finally:
        try:
            conn.close()
        except Exception:
            pass


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
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


threading.Thread(target=serve, daemon=True).start()
Path("/tmp/stale_broker_q_ready").write_text(
    f"READY sid={SID} sock={SOCK} log={LOG} pid={PID}\n", encoding="utf-8")
sys.stderr.write(f"STALE_BROKER_Q_READY sid={SID} sock={SOCK} log={LOG} pid={PID}\n")
sys.stderr.flush()
while True:
    time.sleep(1)
