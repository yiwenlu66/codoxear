#!/usr/bin/env python3
from __future__ import annotations
import json, os, socket, sys, threading, time
from pathlib import Path

HOME = Path(os.environ.get("HOME", "/home/tester"))
APP = HOME / ".local/share/codoxear"
SOCKS = APP / "socks"
CLAUDE = HOME / ".claude"
PROJECTS = CLAUDE / "projects"
CWD = HOME / "mobile-shell-touch-repo"
CALLS = HOME / "mobile-shell-touch-calls.jsonl"
SID = "mobile-shell-touch"
THREAD = "22222222-3333-4444-8555-666666666666"

for path in (SOCKS, PROJECTS, CWD):
    path.mkdir(parents=True, exist_ok=True)
(CWD / "README.md").write_text("mobile shell touch proof repo\n", encoding="utf-8")
log_dir = PROJECTS / "-home-tester-mobile-shell-touch-repo"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"{THREAD}.jsonl"
rows = [
    {
        "type": "user",
        "sessionId": THREAD,
        "timestamp": "2026-07-07T06:30:00.000Z",
        "cwd": str(CWD),
        "message": {"role": "user", "content": "mobile shell touch target proof fixture"},
    }
]
log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

sock_path = SOCKS / f"{SID}.sock"
sidecar = {
    "agent_backend": "cc",
    "session_id": SID,
    "thread_id": THREAD,
    "broker_pid": os.getpid(),
    "codex_pid": os.getpid(),
    "claude_pid": os.getpid(),
    "cwd": str(CWD),
    "log_path": str(log_path),
    "start_ts": time.time(),
    "owner": "terminal",
    "sock_path": str(sock_path),
    "control_protocol_version": 2,
    "control_capabilities": {"sync_send": True, "key_write_errors": False},
}
(SOCKS / f"{SID}.json").write_text(json.dumps(sidecar), encoding="utf-8")
state = {"busy": True, "queue_len": 0, "token": None, "interrupted_idle": False}
if sock_path.exists():
    sock_path.unlink()
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(str(sock_path))
srv.listen(8)
srv.settimeout(0.5)

def log(req: dict, resp: dict) -> None:
    with CALLS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), "req": req, "resp": resp}) + "\n")

def send(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))

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
                resp = {"tail": "mobile-shell fake busy tail"}
            elif cmd == "send":
                resp = {"queued": False, "queue_len": 0, "busy": True}
            elif cmd == "keys":
                resp = {"ok": True, "queued": False, "n": len(str(req.get("seq") or "")), "key_queue_len": 0}
            elif cmd == "shutdown":
                resp = {"ok": True}
            else:
                resp = {"error": "unknown cmd"}
            log(req, resp)
            send(conn, resp)
        except Exception as exc:
            print("fake broker error", exc, file=sys.stderr)
        finally:
            try:
                conn.close()
            except Exception:
                pass

threading.Thread(target=loop, daemon=True).start()
print(json.dumps({"sid": SID, "thread": THREAD, "cwd": str(CWD), "log": str(log_path), "calls": str(CALLS), "sock": str(sock_path)}), flush=True)
while True:
    time.sleep(1)
