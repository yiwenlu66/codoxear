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
SID = "export-too-large-session"
SOCK = SOCKS / f"{SID}.sock"
BASE = HOME / "oversized-export-proof"
LOG = BASE / "oversized-session.jsonl"
CWD = BASE / "workspace"
CALLS = BASE / "broker-calls.jsonl"


def write_files() -> None:
    SOCKS.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    CWD.mkdir(parents=True, exist_ok=True)
    (CWD / "README.md").write_text("oversized export browser proof\n", encoding="utf-8")
    rows = [
        {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hello oversized export"}]}},
        {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "copy should be too large"}], "stopReason": "stop"}},
    ]
    with LOG.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        # Runtime-only padding to exceed the lowered Docker export cap (1024 bytes).
        f.write("\n" * 2048)
    sidecar = {
        "agent_backend": "codex",
        "session_id": SID,
        "thread_id": SID,
        "broker_pid": os.getpid(),
        "codex_pid": os.getpid(),
        "pid": os.getpid(),
        "cwd": str(CWD),
        "log_path": str(LOG),
        "start_ts": time.time(),
        "updated_ts": time.time(),
        "owner": "terminal",
        "sock_path": str(SOCK),
        "model": "gpt-export-proof",
        "reasoning_effort": "high",
        "control_protocol_version": 2,
        "control_capabilities": {"sync_send": True, "key_write_errors": False},
        "fake_notice": "FAKE_OVERSIZED_EXPORT_COPY_PROOF_DOCKER_ONLY",
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
    print(json.dumps({"sid": SID, "sock": str(SOCK), "log": str(LOG), "log_size": LOG.stat().st_size, "calls": str(CALLS)}, separators=(",", ":")), flush=True)
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
            print(f"fake oversized export broker error: {exc}", file=sys.stderr, flush=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass

if __name__ == "__main__":
    serve()
