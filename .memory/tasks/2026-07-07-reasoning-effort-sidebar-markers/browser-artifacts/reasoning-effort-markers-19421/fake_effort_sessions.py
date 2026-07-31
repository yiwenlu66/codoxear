#!/usr/bin/env python3
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
BASE_CWD = HOME / "reasoning-effort-proof"
CALLS = HOME / "reasoning-effort-proof-calls.jsonl"
FAKE_NOTICE = "FAKE_REASONING_EFFORT_MARKER_PROOF_FOR_CODOXEAR_DOCKER_ONLY"

SESSIONS = [
    {"sid": "effort-cc-max", "backend": "cc", "effort": "max", "model": "sonnet", "label": "cc max marker repo"},
    {"sid": "effort-pi-minimal", "backend": "pi", "effort": "minimal", "model": "pi-minimal", "label": "pi minimal marker repo"},
    {"sid": "effort-pi-off", "backend": "pi", "effort": "off", "model": "pi-off", "label": "pi off marker repo"},
    {"sid": "effort-codex-xhigh", "backend": "codex", "effort": "xhigh", "model": "gpt-xhigh", "label": "codex xhigh marker repo"},
    {"sid": "effort-codex-high", "backend": "codex", "effort": "high", "model": "gpt-high", "label": "codex high marker repo"},
    {"sid": "effort-codex-medium", "backend": "codex", "effort": "medium", "model": "gpt-medium", "label": "codex medium marker repo"},
    {"sid": "effort-codex-low", "backend": "codex", "effort": "low", "model": "gpt-low", "label": "codex low marker repo"},
]


def write_sidecars() -> None:
    SOCKS.mkdir(parents=True, exist_ok=True)
    BASE_CWD.mkdir(parents=True, exist_ok=True)
    base_ts = 1783420000.0
    for idx, item in enumerate(SESSIONS):
        sid = item["sid"]
        cwd = BASE_CWD / sid
        cwd.mkdir(parents=True, exist_ok=True)
        (cwd / "README.md").write_text(f"{item['label']}\n", encoding="utf-8")
        sock_path = SOCKS / f"{sid}.sock"
        sidecar = {
            "agent_backend": item["backend"],
            "session_id": sid,
            "thread_id": sid,
            "broker_pid": os.getpid(),
            "codex_pid": os.getpid(),
            "pid": os.getpid(),
            "cwd": str(cwd),
            "log_path": None,
            "start_ts": base_ts + idx,
            "updated_ts": base_ts + idx,
            "owner": "terminal",
            "sock_path": str(sock_path),
            "model": item["model"],
            "reasoning_effort": item["effort"],
            "control_protocol_version": 2,
            "control_capabilities": {"sync_send": True, "key_write_errors": False},
            "fake_notice": FAKE_NOTICE,
        }
        (SOCKS / f"{sid}.json").write_text(json.dumps(sidecar, separators=(",", ":")), encoding="utf-8")


def log_call(sid: str, req: dict, resp: dict) -> None:
    with CALLS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"sid": sid, "ts": time.time(), "req": req, "resp": resp}, separators=(",", ":")) + "\n")


def socket_loop(item: dict) -> None:
    sid = item["sid"]
    sock_path = SOCKS / f"{sid}.sock"
    if sock_path.exists():
        sock_path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(16)
    srv.settimeout(0.5)
    state = {"busy": False, "queue_len": 0, "token": None, "interrupted_idle": False}
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
            log_call(sid, req, resp)
            conn.sendall((json.dumps(resp, separators=(",", ":")) + "\n").encode("utf-8"))
        except Exception as exc:
            print(f"fake effort broker {sid} error: {exc}", file=sys.stderr, flush=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass


def serve() -> None:
    write_sidecars()
    for item in SESSIONS:
        threading.Thread(target=socket_loop, args=(item,), daemon=True).start()
    print(json.dumps({"sessions": SESSIONS, "calls": str(CALLS), "socks": str(SOCKS), "fake_notice": FAKE_NOTICE}, separators=(",", ":")), flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "paths":
        print(json.dumps({"calls": str(CALLS), "socks": str(SOCKS), "sessions": SESSIONS}, separators=(",", ":")))
    else:
        serve()
