#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import socket
import sys
import time
from pathlib import Path

HOME = Path(os.environ.get("HOME", "/home/tester"))
APP = HOME / ".local/share/codoxear"
SOCKS = APP / "socks"
SID = "confirm-focus-session"
SOCK = SOCKS / f"{SID}.sock"
BASE = HOME / "confirm-focus-proof"
WORKSPACE = BASE / "workspace"
PROOF_FILE = WORKSPACE / "proof.txt"
LOG = BASE / "confirm-focus-session.jsonl"
CALLS = BASE / "broker-calls.jsonl"
NOTICE = "FAKE_CONFIRM_FOCUS_PROOF_DOCKER_ONLY"


def write_files() -> None:
    SOCKS.mkdir(parents=True, exist_ok=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not PROOF_FILE.exists():
        PROOF_FILE.write_text("original file text\n", encoding="utf-8")
    rows = [
        {"type": "event_msg", "payload": {"type": "user_message", "message": "confirm focus proof"}, "ts": 1.0},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Proof session for confirmation focus."}], "phase": "final_answer"}, "ts": 2.0},
        {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": "confirm-focus-proof", "last_agent_message": None}, "ts": 3.0},
    ]
    with LOG.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    sidecar = {
        "agent_backend": "codex",
        "session_id": SID,
        "thread_id": SID,
        "broker_pid": os.getpid(),
        "codex_pid": os.getpid(),
        "pid": os.getpid(),
        "cwd": str(WORKSPACE),
        "log_path": str(LOG),
        "start_ts": time.time(),
        "updated_ts": time.time(),
        "owner": "terminal",
        "sock_path": str(SOCK),
        "model": "gpt-confirm-focus-proof",
        "reasoning_effort": "high",
        "control_protocol_version": 2,
        "control_capabilities": {"sync_send": True, "key_write_errors": False},
        "fake_notice": NOTICE,
    }
    (SOCKS / f"{SID}.json").write_text(json.dumps(sidecar, separators=(",", ":")), encoding="utf-8")


def log_call(req: dict, resp: dict) -> None:
    with CALLS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "req": req, "resp": resp}, separators=(",", ":")) + "\n")


def serve() -> None:
    write_files()
    if SOCK.exists():
        SOCK.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK))
    srv.listen(16)
    srv.settimeout(0.5)
    state = {"busy": False, "queue_len": 0, "token": None, "interrupted_idle": False}
    print(json.dumps({"sid": SID, "sock": str(SOCK), "cwd": str(WORKSPACE), "file": str(PROOF_FILE), "log": str(LOG), "calls": str(CALLS), "fake_notice": NOTICE}, separators=(",", ":")), flush=True)
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            line = conn.makefile("rb").readline()
            req = json.loads(line.decode("utf-8") or "{}") if line else {}
            cmd = req.get("cmd")
            if cmd == "state":
                resp = dict(state)
            elif cmd == "tail":
                resp = {"tail": ""}
            elif cmd == "send":
                resp = {"queued": False, "queue_len": 0, "busy": False}
            elif cmd == "keys":
                resp = {"ok": True, "queued": False, "n": len(str(req.get("seq") or "")), "key_queue_len": 0}
            elif cmd == "shutdown":
                resp = {"ok": True}
            else:
                resp = {"error": "unknown cmd"}
            log_call(req, resp)
            conn.sendall((json.dumps(resp, separators=(",", ":")) + "\n").encode("utf-8"))
        except Exception as exc:
            print(f"fake confirm focus broker error: {exc}", file=sys.stderr, flush=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    serve()
