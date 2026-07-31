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
BASE_CWD = HOME / "sidebar-model-proof"
CALLS = HOME / "sidebar-model-proof-calls.jsonl"
FAKE_NOTICE = "FAKE_SIDEBAR_MODEL_DISPLAY_PROOF_FOR_CODOXEAR_DOCKER_ONLY"
LONG_MODEL = "provider/very-long-model-name-for-mobile-ellipsis-proof-claude-sonnet-4-5-extra-suffix"

SESSIONS = [
    {"sid": "model-codex-gpt", "backend": "codex", "model": "gpt-5.4", "effort": "high", "cwd_name": "same-project"},
    {"sid": "model-cc-sonnet", "backend": "cc", "model": "claude-sonnet-4-5", "effort": "max", "cwd_name": "same-project"},
    {"sid": "model-pi-long", "backend": "pi", "model": LONG_MODEL, "effort": "minimal", "cwd_name": "same-project"},
    {"sid": "model-default-omitted", "backend": "codex", "model": "default", "effort": "medium", "cwd_name": "default-model-project"},
    {"sid": "model-empty-omitted", "backend": "pi", "model": "", "effort": "off", "cwd_name": "empty-model-project"},
]


def write_sidecars() -> None:
    SOCKS.mkdir(parents=True, exist_ok=True)
    BASE_CWD.mkdir(parents=True, exist_ok=True)
    base_ts = 1783423000.0
    for idx, item in enumerate(SESSIONS):
        sid = item["sid"]
        cwd = BASE_CWD / item["cwd_name"] / sid
        cwd.mkdir(parents=True, exist_ok=True)
        (cwd / "README.md").write_text(f"{sid}\n", encoding="utf-8")
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
            "reasoning_effort": item["effort"],
            "control_protocol_version": 2,
            "control_capabilities": {"sync_send": True, "key_write_errors": False},
            "fake_notice": FAKE_NOTICE,
        }
        # Preserve the empty-model case as an explicit empty sidecar value. The
        # server should normalize it away and the browser should omit it.
        sidecar["model"] = item["model"]
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
            print(f"fake model broker {sid} error: {exc}", file=sys.stderr, flush=True)
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
