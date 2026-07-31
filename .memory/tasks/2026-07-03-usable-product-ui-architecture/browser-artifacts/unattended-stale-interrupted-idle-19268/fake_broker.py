#!/usr/bin/env python3
"""Stale fake broker for the UNATTED stale-interrupted-idle discriminator.

Reports a PERMANENT stale ``interrupted_idle=true`` (busy:false, queue_len:0)
so the listing/log-watcher suppression path clears the stored override while
the raw broker state still carries ``interrupted_idle:true``.

The discriminator log isolates the UNATTED readiness gate as the sole blocker:

  baseline (interrupted turn): user_message + non-final assistant fragment
  appended post-interrupt:     task_complete(old ts, last_agent_message="done")
                               agent_reasoning(later ts)

Then in current code:
  * ``_compute_idle_from_log``            -> False (latest agent_reasoning is busy)
  * ``_last_chat_role_ts_from_tail(final_assistant_only=True)``
                                          -> ("assistant", task_complete_ts)
    because agent_reasoning is NOT a chat role, so the unattended tail gate
    PASSES (assistant, old ts >= cooldown). The ONLY thing that can stop the
    sweep from injecting is the readiness gate using the session-authoritative
    (suppressed) interrupted_idle. If readiness reactivates the stale raw
    broker override, the sweep injects -> DEFECT.

Logs EVERY command received to /tmp/unattended_broker_calls.jsonl so the driver
can prove whether the server's real unattended sweep attempted a confirmed send
(``cmd:send``) or key-write (``cmd:keys``) while the sidebar reported busy.

Implements ``cmd:state``, ``cmd:tail``, ``cmd:send`` (confirmed sync send),
``cmd:keys``, ``cmd:shutdown``. Writes socket + sidecar + initial interrupted-
turn Codex rollout log under the container app dir so the REAL server discovers
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
SID = "cert-unattended-stale"
LOG = LOGDIR / f"{SID}.jsonl"
SOCK = SOCKS / f"{SID}.sock"
SIDECAR = SOCKS / f"{SID}.json"
CALLLOG = Path("/tmp/unattended_broker_calls.jsonl")


def write_initial_log():
    """Interrupted turn: user + non-final assistant fragment (no completion).

    ``_compute_idle_from_log`` returns False for this tail (non-final
    assistant). With a valid raw broker ``interrupted_idle:true`` override the
    listing projects idle until post-interrupt activity lands.
    """
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


write_initial_log()

sidecar = {
    "agent_backend": "codex", "session_id": SID, "thread_id": SID,
    "broker_pid": PID, "codex_pid": PID, "cwd": "/home/tester",
    "log_path": str(LOG), "start_ts": time.time(), "owner": "terminal",
    "sock_path": str(SOCK), "control_protocol_version": 2,
    "control_capabilities": {"sync_send": True, "key_write_errors": True},
}
SIDECAR.write_text(json.dumps(sidecar), encoding="utf-8")

# Reset call log.
CALLLOG.write_text("", encoding="utf-8")
CALL_LOCK = threading.Lock()


def record_call(cmd: str, req: dict, resp: dict) -> None:
    entry = {"ts": time.time(), "cmd": cmd, "req": req, "resp": resp}
    with CALL_LOCK:
        with CALLLOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


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
            send_json(conn, resp)
            record_call(cmd, {"cmd": "state"}, resp)
        elif cmd == "tail":
            resp = {"tail": ""}
            send_json(conn, resp)
            record_call(cmd, {"cmd": "tail"}, resp)
        elif cmd == "send":
            # Confirmed sync send: respond with a well-formed acceptance so
            # parse_confirmed_send_response succeeds. The discriminator cares
            # ONLY that the server's unattended sweep reached this point.
            resp = {"queued": False, "queue_len": 0, "busy": True}
            send_json(conn, resp)
            record_call(cmd, {"cmd": "send", "text": req.get("text"),
                              "sync": req.get("sync"),
                              "queue_item_id": req.get("queue_item_id")}, resp)
        elif cmd == "keys":
            resp = {"ok": True, "queued": False, "n": 0, "key_queue_len": 0}
            send_json(conn, resp)
            record_call(cmd, {"cmd": "keys", "seq": req.get("seq")}, resp)
        elif cmd == "shutdown":
            resp = {"ok": True}
            send_json(conn, resp)
            record_call(cmd, {"cmd": "shutdown"}, resp)
        else:
            resp = {"error": "unknown cmd"}
            send_json(conn, resp)
            record_call(cmd, {"cmd": cmd}, resp)
    except Exception as e:
        sys.stderr.write(f"broker error: {e}\n")
        sys.stderr.flush()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def serve():
    if SOCK.exists():
        SOCK.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK))
    srv.listen(8)
    srv.settimeout(0.5)
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


threading.Thread(target=serve, daemon=True).start()
Path("/tmp/unattended_broker_ready").write_text(
    f"READY sid={SID} sock={SOCK} log={LOG} pid={PID}\n", encoding="utf-8")
sys.stderr.write(f"UNATTED_BROKER_READY sid={SID} sock={SOCK} log={LOG} pid={PID}\n")
sys.stderr.flush()
while True:
    time.sleep(1)
