#!/usr/bin/env python3
"""Deterministic fake-broker harness for Codoxear certification.

Creates fake session control sockets + sidecars + Codex rollout logs inside the
container app dir so the running server discovers them. Each fake broker
responds to the control-socket protocol (state / send / keys / shutdown) with
newline-delimited JSON, exactly like codoxear.broker_control.

This is EVIDENCE ONLY: it does not edit product code. It fabricates the
session/log/upload state the task explicitly permits so the API and browser
can be exercised deterministically without real backend credentials.
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
UPLOADS = APP / "uploads"
LOGDIR = HOME / "cert-logs"
SOCKS.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

PID = os.getpid()
START_TS = time.time()


def send_json_line(conn, obj):
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def socket_server(sock_path: Path, state: dict, shutdown_event: threading.Event,
                  on_shutdown=None):
    """Listen on a Unix socket; answer one request per connection."""
    if sock_path.exists():
        sock_path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(8)
    srv.settimeout(0.5)

    def loop():
        while not shutdown_event.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                f = conn.makefile("rb")
                line = f.readline()
                if not line:
                    continue
                req = json.loads(line.decode("utf-8"))
                cmd = req.get("cmd")
                if cmd == "state":
                    send_json_line(conn, dict(state))
                elif cmd == "tail":
                    send_json_line(conn, {"tail": ""})
                elif cmd == "send":
                    # Pretend the send was committed synchronously.
                    send_json_line(conn, {"queued": False, "queue_len": 0})
                elif cmd == "keys":
                    send_json_line(conn, {"ok": True, "queued": False, "n": 0,
                                          "key_queue_len": 0})
                elif cmd == "shutdown":
                    send_json_line(conn, {"ok": True})
                    if on_shutdown:
                        on_shutdown()
                else:
                    send_json_line(conn, {"error": "unknown cmd"})
            except Exception as e:
                sys.stderr.write(f"socket_server error on {sock_path}: {e}\n")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        try:
            srv.close()
        except Exception:
            pass

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def write_log(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def make_sidecar(sid: str, log_path: Path, agent_backend: str = "codex",
                 owner: str = "terminal") -> dict:
    return {
        "agent_backend": agent_backend,
        "session_id": sid,
        "broker_pid": PID,
        "codex_pid": PID,
        "cwd": str(HOME),
        "log_path": str(log_path),
        "start_ts": START_TS,
        "owner": owner,
        "sock_path": str(SOCKS / f"{sid}.sock"),
        # Advertise control protocol v2 with sync_send + key_write_errors so the
        # server allows confirmed sends and file-attachment injection against
        # this fake broker (mirrors what codoxear-broker publishes).
        "control_protocol_version": 2,
        "control_capabilities": {"sync_send": True, "key_write_errors": True},
    }


def make_session(sid: str, rows: list[dict], agent_backend: str = "codex",
                 busy: bool = False, interrupted_idle: bool = False, cwd: str | None = None):
    log_path = write_log(LOGDIR / f"{sid}.jsonl", rows)
    sidecar = make_sidecar(sid, log_path, agent_backend=agent_backend)
    if cwd:
        sidecar["cwd"] = cwd
    (SOCKS / f"{sid}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    state = {
        "busy": busy,
        "queue_len": 0,
        "token": None,
        "interrupted_idle": interrupted_idle,
    }
    shutdown_evt = threading.Event()
    socket_server(SOCKS / f"{sid}.sock", state, shutdown_evt)
    return {"sid": sid, "log_path": log_path, "sidecar": sidecar,
            "state": state, "shutdown_event": shutdown_evt}


# --- Session fixtures -----------------------------------------------------

NO_RESP_TEXT = "The backend completed this turn without producing a response."

# Claim 4: Codex turn with user input but NO assistant output, then task_complete.
noresp_rows = [
    {"type": "event_msg", "ts": 1.0,
     "payload": {"type": "user_message", "message": "please summarize the report"}},
    {"type": "event_msg", "ts": 2.0,
     "payload": {"type": "task_complete", "turn_id": "t1", "last_agent_message": None}},
]
s_noresp = make_session("cert-noresp", noresp_rows, cwd="/workspace/noresp")

# Control: a normal answered Codex turn (must NOT emit no-response).
normal_rows = [
    {"type": "event_msg", "ts": 1.0,
     "payload": {"type": "user_message", "message": "hello"}},
    {"type": "response_item", "ts": 2.0, "payload": {
        "type": "message", "role": "assistant", "phase": "final_answer",
        "content": [{"type": "output_text", "text": "world"}]}},
    {"type": "event_msg", "ts": 3.0,
     "payload": {"type": "task_complete", "turn_id": "t1"}},
]
s_normal = make_session("cert-normal", normal_rows, cwd="/workspace/normal")

# Claim 2: a session to delete (for upload cleanup). Staged upload bytes will
# live under uploads/cert-cleanup/.
cleanup_rows = [
    {"type": "event_msg", "ts": 1.0,
     "payload": {"type": "user_message", "message": "attach and delete me"}},
]
s_cleanup = make_session("cert-cleanup", cleanup_rows, cwd="/workspace/cleanup")

# Claim 5: a "resumed after interrupt" session: busy False, interrupted_idle
# False — i.e. NOT falsely idle. We model a resumed/idle-healthy session.
s_resume = make_session("cert-resume", [
    {"type": "event_msg", "ts": 1.0,
     "payload": {"type": "user_message", "message": "resume after interrupt"}},
    {"type": "event_msg", "ts": 2.0,
     "payload": {"type": "task_complete", "turn_id": "t1"}},
], busy=False, interrupted_idle=False, cwd="/workspace/resume")

# Claim 1: clean idle sessions dedicated to attachment-indicator testing.
# Two are used so the send-path and clear-path sub-tests each start from a
# fresh (never-sent, not-busy) session: a send permanently flips the log-idle
# heuristic to busy, which would block a later inject_file on the same session.
s_attach = make_session("cert-attach-a", [
    {"type": "event_msg", "ts": 1.0,
     "payload": {"type": "user_message", "message": "ready for attachment test A"}},
    {"type": "event_msg", "ts": 2.0,
     "payload": {"type": "task_complete", "turn_id": "t1"}},
], busy=False, interrupted_idle=False, cwd="/workspace/attachA")
s_attach_b = make_session("cert-attach-b", [
    {"type": "event_msg", "ts": 1.0,
     "payload": {"type": "user_message", "message": "ready for attachment test B"}},
    {"type": "event_msg", "ts": 2.0,
     "payload": {"type": "task_complete", "turn_id": "t1"}},
], busy=False, interrupted_idle=False, cwd="/workspace/attachB")

# --- Upload fixtures for cleanup claim -----------------------------------

# Target session upload dir with a staged file.
(UPLOADS / "cert-cleanup").mkdir(parents=True, exist_ok=True)
(UPLOADS / "cert-cleanup" / "report.pdf").write_bytes(b"%PDF-1.4 fake staged attachment")

# Sibling session upload dir that must survive the delete of cert-cleanup.
(UPLOADS / "cert-sibling").mkdir(parents=True, exist_ok=True)
(UPLOADS / "cert-sibling" / "keep.txt").write_bytes(b"sibling staged file must remain")

# A symlink entry inside uploads whose target is OUTSIDE uploads; it must NOT
# be removed when cert-cleanup is deleted (different session id), and even if
# it were the target session, the link itself is unlinked, never followed.
secret = HOME / "secret-outside-uploads.txt"
secret.write_bytes(b"host secret outside uploads - must survive")
try:
    link = UPLOADS / "cert-outside-link"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(secret)
except OSError as e:
    sys.stderr.write(f"symlink setup note: {e}\n")

sys.stderr.write(
    "FAKE_BROKER_READY sids="
    f"{[s_noresp['sid'], s_normal['sid'], s_cleanup['sid'], s_resume['sid'], s_attach['sid'], s_attach_b['sid']]}\n"
)
sys.stderr.write(f"UPLOADS={UPLOADS}\n")
sys.stderr.flush()

# Keep alive so the server can discover + query the sockets.
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
