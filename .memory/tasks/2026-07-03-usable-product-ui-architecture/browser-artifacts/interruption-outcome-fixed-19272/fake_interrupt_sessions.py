#!/usr/bin/env python3
"""Synthetic interrupted-turn sessions for the interruption-outcome FIXED proof.

Writes three deterministic synthetic sessions (logs + sidecar metadata + live
broker control sockets) under the throwaway container HOME so the REAL
codoxear.server discovery + /api/messages (tail + search) surfaces project them.

Three scenarios (matching the committed defect proof's input rows, which are
known to trigger the abort path):

  interrupt-pi-empty    Pi user message, then assistant stopReason:"aborted"
                        with empty content.
  interrupt-pi-partial  Pi user message, then assistant stopReason:"aborted"
                        with partial text "I was halfway through the answer".
  interrupt-codex-abort Codex event_msg user_message, then event_msg turn_aborted.

Read-only with respect to source: imports nothing from codoxear. Writes only
its own synthetic state under $HOME/.local/share/codoxear inside the container.
The helper process stays alive so (a) the bound unix sockets remain reachable
for the discovery 'state' call and (b) the sidecar broker/codex PIDs are alive.
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
LOGS = APP / "interrupt-logs"
HELPER_PID = os.getpid()
SOCKS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def write_log(sid: str, rows: list[dict]) -> Path:
    path = LOGS / f"{sid}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


# --- Pi log rows (same shape as the committed defect proof) -----------------
def pi_user_row(ts: float, text: str) -> dict:
    return {
        "type": "message",
        "ts": ts,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def pi_assistant_aborted_row(ts: float, *, partial_text: str | None) -> dict:
    content: list[dict] = []
    if partial_text:
        content.append({"type": "text", "text": partial_text})
    return {
        "type": "message",
        "ts": ts,
        "message": {"role": "assistant", "stopReason": "aborted", "content": content},
    }


# --- Codex log rows ----------------------------------------------------------
def codex_event(payload: dict, ts: float) -> dict:
    return {"type": "event_msg", "ts": ts, "payload": payload}


sessions = {
    "interrupt-pi-empty": {
        "agent_backend": "pi",
        "rows": [
            pi_user_row(1.0, "hello pi (empty abort proof)"),
            pi_assistant_aborted_row(2.0, partial_text=None),
        ],
    },
    "interrupt-pi-partial": {
        "agent_backend": "pi",
        "rows": [
            pi_user_row(1.0, "hello pi (partial abort proof)"),
            pi_assistant_aborted_row(2.0, partial_text="I was halfway through the answer"),
        ],
    },
    "interrupt-codex-abort": {
        "agent_backend": "codex",
        "rows": [
            codex_event({"type": "user_message", "message": "hello codex (turn_aborted proof)"}, 1.0),
            codex_event({"type": "turn_aborted"}, 2.0),
        ],
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
        "agent_backend": spec["agent_backend"],
        "session_id": sid,
        "thread_id": f"thread-{sid}",
        "codex_pid": HELPER_PID,
        "broker_pid": HELPER_PID,
        "cwd": "/workspace",
        "log_path": str(log),
        "start_ts": 100.0,
        "owner": "terminal",
        "source": "interruption-outcome-fixed-proof",
    }
    (SOCKS / f"{sid}.json").write_text(json.dumps(sidecar) + "\n", encoding="utf-8")
    threading.Thread(target=serve, args=(sock,), daemon=True).start()

print(json.dumps({"ready": True, "sessions": sorted(sessions), "app": str(APP), "logs": str(LOGS), "pid": HELPER_PID}), flush=True)
while True:
    time.sleep(60)
