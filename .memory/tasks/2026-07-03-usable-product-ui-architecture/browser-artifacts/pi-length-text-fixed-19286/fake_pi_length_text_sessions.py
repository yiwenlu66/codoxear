#!/usr/bin/env python3
"""Synthetic Pi visible-text length sessions for fixed false-idle verification."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

APP = Path(os.environ.get("HOME", "/home/tester")) / ".local" / "share" / "codoxear"
SOCKS = APP / "socks"
LOGS = APP / "pi-length-text-fixed-logs"
HELPER_PID = os.getpid()
SOCKS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def write_log(sid: str, rows: list[dict]) -> Path:
    path = LOGS / f"{sid}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def session_row(sid: str) -> dict:
    return {"type": "session", "id": sid, "cwd": "/workspace", "timestamp": "2026-01-01T00:00:00Z"}


def user_row(ts: float, text: str) -> dict:
    return {"type": "message", "ts": ts, "timestamp": f"2026-01-01T00:00:0{int(ts)}Z", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}


def assistant_row(ts: float, stop_reason: str, content: list[dict]) -> dict:
    return {"type": "message", "ts": ts, "timestamp": f"2026-01-01T00:00:0{int(ts)}Z", "message": {"role": "assistant", "stopReason": stop_reason, "content": content}}


sessions = {
    "pi-length-text-prefix-fixed": [
        session_row("pi-length-text-prefix-fixed"),
        user_row(1.0, "hello pi length text prefix"),
        assistant_row(2.0, "length", [{"type": "text", "text": "partial before compaction"}]),
    ],
    "pi-length-text-continuation-fixed": [
        session_row("pi-length-text-continuation-fixed"),
        user_row(1.0, "hello pi length text continuation"),
        assistant_row(2.0, "length", [{"type": "text", "text": "partial before compaction"}]),
        {"type": "compaction", "ts": 3.0, "timestamp": "2026-01-01T00:00:03Z", "message": "compacting context"},
        {"type": "custom_message", "ts": 4.0, "timestamp": "2026-01-01T00:00:04Z", "message": "continuing after compaction"},
        assistant_row(5.0, "toolUse", [
            {"type": "text", "text": "resuming after compaction and calling a tool"},
            {"type": "toolCall", "id": "toolu_1", "name": "bash", "arguments": {"command": "pwd"}},
        ]),
    ],
    "pi-stop-text-control": [
        session_row("pi-stop-text-control"),
        user_row(1.0, "hello pi stop text control"),
        assistant_row(2.0, "stop", [{"type": "text", "text": "final answer"}]),
    ],
}


def serve(sock_path: Path) -> None:
    try:
        sock_path.unlink()
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(16)
    srv.settimeout(0.4)

    def handle(conn: socket.socket) -> None:
        try:
            line = conn.makefile("rb").readline()
            req = json.loads(line.decode()) if line else {}
            cmd = req.get("cmd") if isinstance(req, dict) else None
            if cmd == "state":
                resp = {"busy": False, "queue_len": 0, "interrupted_idle": False, "token": None}
            elif cmd == "tail":
                resp = {"tail": ""}
            else:
                resp = {"ok": True}
            conn.sendall((json.dumps(resp) + "\n").encode())
        except Exception as exc:
            try:
                conn.sendall((json.dumps({"error": str(exc)}) + "\n").encode())
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


for sid, rows in sessions.items():
    log = write_log(sid, rows)
    sock = SOCKS / f"{sid}.sock"
    sidecar = {
        "agent_backend": "pi",
        "session_id": sid,
        "thread_id": f"thread-{sid}",
        "codex_pid": HELPER_PID,
        "broker_pid": HELPER_PID,
        "cwd": "/workspace",
        "log_path": str(log),
        "start_ts": 100.0,
        "owner": "terminal",
        "source": "pi-length-text-fixed-proof",
    }
    (SOCKS / f"{sid}.json").write_text(json.dumps(sidecar) + "\n", encoding="utf-8")
    threading.Thread(target=serve, args=(sock,), daemon=True).start()

print(json.dumps({"ready": True, "sessions": sorted(sessions), "app": str(APP), "logs": str(LOGS), "pid": HELPER_PID}), flush=True)
while True:
    time.sleep(60)
