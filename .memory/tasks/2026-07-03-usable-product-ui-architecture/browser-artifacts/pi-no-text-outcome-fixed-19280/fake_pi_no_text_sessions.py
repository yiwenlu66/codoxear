#!/usr/bin/env python3
"""Synthetic Pi no-visible-response sessions for fixed-outcome verification.

Writes deterministic Pi logs + sidecars + live broker control sockets under the
container HOME app dir. The helper stays alive so sidecar PIDs and sockets stay
valid for real codoxear.server discovery.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

APP = Path(os.environ.get("HOME", "/home/tester")) / ".local" / "share" / "codoxear"
SOCKS = APP / "socks"
LOGS = APP / "pi-no-text-logs"
HELPER_PID = os.getpid()
SOCKS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def write_log(sid: str, rows: list[dict]) -> Path:
    path = LOGS / f"{sid}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def pi_session_row(sid: str) -> dict:
    return {"type": "session", "id": sid, "cwd": "/workspace", "timestamp": "2026-01-01T00:00:00Z"}


def pi_user_row(ts: float, text: str) -> dict:
    return {
        "type": "message",
        "ts": ts,
        "timestamp": f"2026-01-01T00:00:0{int(ts)}Z",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def pi_assistant_row(ts: float, stop_reason: str | None, content: list[dict]) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if stop_reason is not None:
        msg["stopReason"] = stop_reason
    return {
        "type": "message",
        "ts": ts,
        "timestamp": f"2026-01-01T00:00:0{int(ts)}Z",
        "message": msg,
    }


sessions = {
    "pi-no-text-stop-empty": {
        "prompt": "hello pi stop empty proof",
        "rows": [
            pi_session_row("pi-no-text-stop-empty"),
            pi_user_row(1.0, "hello pi stop empty proof"),
            pi_assistant_row(2.0, "stop", []),
        ],
        "terminal": True,
    },
    "pi-no-text-end-turn-empty": {
        "prompt": "hello pi end turn empty proof",
        "rows": [
            pi_session_row("pi-no-text-end-turn-empty"),
            pi_user_row(1.0, "hello pi end turn empty proof"),
            pi_assistant_row(2.0, "end_turn", []),
        ],
        "terminal": True,
    },
    "pi-no-text-stop-thinking": {
        "prompt": "hello pi stop thinking proof",
        "rows": [
            pi_session_row("pi-no-text-stop-thinking"),
            pi_user_row(1.0, "hello pi stop thinking proof"),
            pi_assistant_row(2.0, "stop", [{"type": "thinking", "thinking": ""}]),
        ],
        "terminal": True,
    },
    "pi-nonterminal-thinking-control": {
        "prompt": "hello pi nonterminal thinking control",
        "rows": [
            pi_session_row("pi-nonterminal-thinking-control"),
            pi_user_row(1.0, "hello pi nonterminal thinking control"),
            pi_assistant_row(2.0, None, [{"type": "thinking", "thinking": ""}]),
        ],
        "terminal": False,
    },
    "pi-tool-use-control": {
        "prompt": "hello pi tool use control",
        "rows": [
            pi_session_row("pi-tool-use-control"),
            pi_user_row(1.0, "hello pi tool use control"),
            pi_assistant_row(2.0, "toolUse", [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {}}]),
        ],
        "terminal": False,
    },
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


for sid, spec in sessions.items():
    log = write_log(sid, spec["rows"])
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
        "source": "pi-no-text-outcome-fixed-proof",
    }
    (SOCKS / f"{sid}.json").write_text(json.dumps(sidecar) + "\n", encoding="utf-8")
    threading.Thread(target=serve, args=(sock,), daemon=True).start()

print(json.dumps({"ready": True, "sessions": sorted(sessions), "app": str(APP), "logs": str(LOGS), "pid": HELPER_PID}), flush=True)
while True:
    time.sleep(60)
