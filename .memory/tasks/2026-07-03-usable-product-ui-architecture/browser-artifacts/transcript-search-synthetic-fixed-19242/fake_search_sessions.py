#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

APP = Path(os.environ.get("HOME", "/home/tester")) / ".local" / "share" / "codoxear"
SOCKS = APP / "socks"
LOGS = APP / "search-logs"
HELPER_PID = os.getpid()
SOCKS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def write_log(sid: str, rows: list[dict]) -> Path:
    path = LOGS / f"{sid}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def codex_event(payload: dict, ts: float) -> dict:
    return {"type": "event_msg", "ts": ts, "payload": payload}


def cc_user(text: str) -> dict:
    return {
        "type": "user",
        "timestamp": "2026-07-06T00:00:00.000Z",
        "sessionId": "cc-search-session",
        "cwd": "/workspace",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def cc_turn_duration() -> dict:
    return {
        "type": "system",
        "subtype": "turn_duration",
        "timestamp": "2026-07-06T00:00:03.000Z",
        "sessionId": "cc-search-session",
        "durationMs": 1234,
    }


def cc_api_error() -> dict:
    return {
        "type": "system",
        "subtype": "api_error",
        "timestamp": "2026-07-06T00:00:05.000Z",
        "sessionId": "cc-search-session",
        "error": "API Error: 503 Search Proof",
        "retryAttempt": 3,
        "maxRetries": 3,
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


sessions = {
    "search-codex-noresp": {
        "agent_backend": "codex",
        "rows": [
            codex_event({"type": "user_message", "message": "codex silent search prompt"}, 1.0),
            codex_event({"type": "task_complete", "turn_id": "silent", "last_agent_message": None}, 2.0),
        ],
    },
    "search-codex-answered": {
        "agent_backend": "codex",
        "rows": [
            codex_event({"type": "user_message", "message": "codex answered prompt"}, 3.0),
            codex_event({"type": "agent_message", "phase": "final_answer", "message": "CODEX-ANSWER-SEARCH"}, 4.0),
            codex_event({"type": "task_complete", "turn_id": "answered"}, 5.0),
        ],
    },
    "search-cc-noresp": {
        "agent_backend": "cc",
        "rows": [cc_user("cc silent search prompt"), cc_turn_duration()],
    },
    "search-cc-apierr": {
        "agent_backend": "cc",
        "rows": [cc_user("cc failing search prompt"), cc_api_error()],
    },
}

for sid, spec in sessions.items():
    log = write_log(sid, spec["rows"])
    sock = SOCKS / f"{sid}.sock"
    sidecar = {
        "agent_backend": spec["agent_backend"],
        "session_id": sid,
        "thread_id": f"thread-{sid}",
        "codex_pid": HELPER_PID,
        "broker_pid": HELPER_PID,
        "cwd": "/workspace",
        "log_path": str(log),
        "start_ts": 100.0,
        "owner": "terminal",
        "source": "search-synthetic-proof",
    }
    (SOCKS / f"{sid}.json").write_text(json.dumps(sidecar) + "\n", encoding="utf-8")
    threading.Thread(target=serve, args=(sock,), daemon=True).start()

print(json.dumps({"ready": True, "sessions": sorted(sessions), "app": str(APP), "logs": str(LOGS)}), flush=True)
while True:
    time.sleep(60)
