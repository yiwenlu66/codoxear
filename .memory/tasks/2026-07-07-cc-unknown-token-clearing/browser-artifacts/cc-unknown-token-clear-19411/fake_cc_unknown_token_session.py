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
CLAUDE = HOME / ".claude"
PROJECTS = CLAUDE / "projects"
CWD = HOME / "cc-unknown-token-proof-repo"
CALLS = HOME / "cc-unknown-token-proof-calls.jsonl"
SID = "cc-unknown-token-proof"
THREAD = "22222222-3333-4444-8555-666666666666"
LOG_DIR = PROJECTS / "-home-tester-cc-unknown-token-proof-repo"
LOG_PATH = LOG_DIR / f"{THREAD}.jsonl"
SOCK_PATH = SOCKS / f"{SID}.sock"
FAKE_LABEL = "FAKE_CC_UNKNOWN_TOKEN_CLEAR_FOR_CODOXEAR_DOCKER_ONLY"


def known_rows() -> list[dict]:
    return [
        {
            "type": "user",
            "sessionId": THREAD,
            "timestamp": "2026-07-07T09:00:00.000Z",
            "cwd": str(CWD),
            "message": {"role": "user", "content": "cc unknown token clearing fixture"},
            "fake_notice": FAKE_LABEL,
        },
        {
            "type": "assistant",
            "sessionId": THREAD,
            "timestamp": "2026-07-07T09:00:10.000Z",
            "cwd": str(CWD),
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "KNOWN_MODEL_TOKEN_SENTINEL visible before unknown"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 4321,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 79,
                    "output_tokens": 9999,
                },
            },
            "fake_notice": FAKE_LABEL,
        },
    ]


def unknown_row() -> dict:
    return {
        "type": "assistant",
        "sessionId": THREAD,
        "timestamp": "2026-07-07T09:01:00.000Z",
        "cwd": str(CWD),
        "message": {
            "role": "assistant",
            "model": "claude-unmapped-future-9",
            "content": [{"type": "text", "text": "UNKNOWN_MODEL_CLEAR_SENTINEL should hide context chip"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12000,
                "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 400,
                "output_tokens": 1,
            },
        },
        "fake_notice": FAKE_LABEL,
    }


def prepare_files() -> None:
    for p in (SOCKS, PROJECTS, CWD, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)
    (CWD / "README.md").write_text("cc unknown token clearing proof repo\n", encoding="utf-8")
    LOG_PATH.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in known_rows()), encoding="utf-8")
    sidecar = {
        "agent_backend": "cc",
        "session_id": SID,
        "thread_id": THREAD,
        "broker_pid": os.getpid(),
        "codex_pid": os.getpid(),
        "claude_pid": os.getpid(),
        "cwd": str(CWD),
        "log_path": str(LOG_PATH),
        "start_ts": time.time(),
        "owner": "terminal",
        "sock_path": str(SOCK_PATH),
        "model": "sonnet",
        "reasoning_effort": "max",
        "control_protocol_version": 2,
        "control_capabilities": {"sync_send": True, "key_write_errors": False},
        "fake_notice": FAKE_LABEL,
    }
    (SOCKS / f"{SID}.json").write_text(json.dumps(sidecar, separators=(",", ":")), encoding="utf-8")


def append_unknown() -> None:
    if not LOG_PATH.exists():
        raise SystemExit(f"missing log: {LOG_PATH}")
    existing = LOG_PATH.read_text(encoding="utf-8")
    if "UNKNOWN_MODEL_CLEAR_SENTINEL" not in existing:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(unknown_row(), separators=(",", ":")) + "\n")
    print(json.dumps({"appended": True, "log_path": str(LOG_PATH), "size": LOG_PATH.stat().st_size}, separators=(",", ":")))


def log_call(req: dict, resp: dict) -> None:
    with CALLS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "req": req, "resp": resp}, separators=(",", ":")) + "\n")


def serve() -> None:
    prepare_files()
    if SOCK_PATH.exists():
        SOCK_PATH.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK_PATH))
    srv.listen(8)
    srv.settimeout(0.5)
    state = {"busy": False, "queue_len": 0, "token": None, "interrupted_idle": False}

    def loop() -> None:
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
                print(f"fake broker error: {exc}", file=sys.stderr, flush=True)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    threading.Thread(target=loop, daemon=True).start()
    print(json.dumps({"sid": SID, "thread": THREAD, "cwd": str(CWD), "log": str(LOG_PATH), "calls": str(CALLS), "sock": str(SOCK_PATH), "fake_notice": FAKE_LABEL}, separators=(",", ":")), flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if mode == "serve":
        serve()
    elif mode == "append-unknown":
        append_unknown()
    elif mode == "paths":
        print(json.dumps({"sid": SID, "thread": THREAD, "log": str(LOG_PATH), "sock": str(SOCK_PATH), "calls": str(CALLS)}, separators=(",", ":")))
    else:
        raise SystemExit(f"unknown mode: {mode}")
