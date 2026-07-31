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
ROOT = HOME / "cc-context-chip-accessible-proof"
CALLS = HOME / "cc-context-chip-accessible-calls.jsonl"
SUMMARY = HOME / "cc-context-chip-accessible-call-summary.json"

VISIBLE_SID = "cc-context-chip-visible"
VISIBLE_THREAD = "22222222-3333-4444-8555-666666666666"
NO_TOKEN_SID = "cc-context-chip-no-token"
NO_TOKEN_THREAD = "33333333-4444-4555-8666-777777777777"
MODEL = "claude-sonnet-4-5"

for p in (SOCKS, PROJECTS, ROOT):
    p.mkdir(parents=True, exist_ok=True)
(ROOT / "README.md").write_text("cc context chip accessibility proof repo\n", encoding="utf-8")

COUNTS = {"send_count": 0, "key_count": 0, "by_session": {}}
COUNT_LOCK = threading.Lock()
SERVERS: list[socket.socket] = []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def log_dir_for(cwd: Path) -> Path:
    return PROJECTS / str(cwd).replace("/", "-")


def make_session(*, sid: str, thread: str, token: bool) -> dict:
    cwd = ROOT / sid
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "README.md").write_text(f"fixture for {sid}\n", encoding="utf-8")
    log_dir = log_dir_for(cwd)
    log_path = log_dir / f"{thread}.jsonl"
    rows = [
        {
            "type": "user",
            "sessionId": thread,
            "timestamp": "2026-07-07T06:10:00.000Z",
            "cwd": str(cwd),
            "message": {"role": "user", "content": f"fixture {sid}"},
        },
        {
            "type": "assistant",
            "sessionId": thread,
            "timestamp": "2026-07-07T06:10:05.000Z",
            "message": {
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": "ready"}],
                "stop_reason": "end_turn",
            },
        },
        {"type": "system", "subtype": "turn_duration", "sessionId": thread, "timestamp": "2026-07-07T06:10:06.000Z", "durationMs": 1000},
    ]
    if token:
        # The frontend chip must project prompt-side context only. For the mapped
        # 200000-token Claude window, this yields maxInput=180000,
        # reserved=20000, used=150000, and percent remaining=17.
        rows[1]["message"]["usage"] = {
            "input_tokens": 100000,
            "cache_read_input_tokens": 20000,
            "cache_creation_input_tokens": 30000,
            "output_tokens": 9999,
            "service_tier": "standard",
        }
    write_jsonl(log_path, rows)
    sock_path = SOCKS / f"{sid}.sock"
    sidecar = {
        "agent_backend": "cc",
        "session_id": sid,
        "thread_id": thread,
        "broker_pid": os.getpid(),
        "codex_pid": os.getpid(),
        "claude_pid": os.getpid(),
        "cwd": str(cwd),
        "log_path": str(log_path),
        "start_ts": time.time() + (1 if token else 0),
        "owner": "terminal",
        "sock_path": str(sock_path),
        "control_protocol_version": 2,
        "control_capabilities": {"sync_send": True, "key_write_errors": False},
    }
    (SOCKS / f"{sid}.json").write_text(json.dumps(sidecar, separators=(",", ":")), encoding="utf-8")
    return {"sid": sid, "thread": thread, "cwd": str(cwd), "log": str(log_path), "sock": str(sock_path), "token": token}


def write_summary() -> None:
    with COUNT_LOCK:
        SUMMARY.write_text(json.dumps(COUNTS, indent=2, sort_keys=True), encoding="utf-8")


def append_call(sid: str, req: dict, resp: dict) -> None:
    with COUNT_LOCK:
        cmd = req.get("cmd")
        per = COUNTS["by_session"].setdefault(sid, {"send_count": 0, "key_count": 0, "state_count": 0, "tail_count": 0, "other_count": 0})
        if cmd == "send":
            COUNTS["send_count"] += 1
            per["send_count"] += 1
        elif cmd == "keys":
            COUNTS["key_count"] += 1
            per["key_count"] += 1
        elif cmd == "state":
            per["state_count"] += 1
        elif cmd == "tail":
            per["tail_count"] += 1
        else:
            per["other_count"] += 1
        with CALLS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "session_id": sid, "req": req, "resp": resp}, separators=(",", ":")) + "\n")
    write_summary()


def start_broker(session: dict) -> None:
    sid = session["sid"]
    state = {"busy": False, "queue_len": 0, "token": None, "interrupted_idle": False}
    sock_path = Path(session["sock"])
    if sock_path.exists():
        sock_path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(16)
    srv.settimeout(0.5)
    SERVERS.append(srv)

    def send(conn: socket.socket, obj: dict) -> None:
        conn.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))

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
                    resp = {"queued": False, "queue_len": 0, "busy": True}
                elif cmd == "keys":
                    resp = {"ok": True, "queued": False, "n": len(str(req.get("seq") or "")), "key_queue_len": 0}
                elif cmd == "shutdown":
                    resp = {"ok": True}
                else:
                    resp = {"error": "unknown cmd"}
                append_call(sid, req, resp)
                send(conn, resp)
            except Exception as exc:
                print(f"fake broker error for {sid}: {exc}", file=sys.stderr, flush=True)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    threading.Thread(target=loop, daemon=True).start()


def main() -> None:
    if CALLS.exists():
        CALLS.unlink()
    visible = make_session(sid=VISIBLE_SID, thread=VISIBLE_THREAD, token=True)
    no_token = make_session(sid=NO_TOKEN_SID, thread=NO_TOKEN_THREAD, token=False)
    for sess in (visible, no_token):
        start_broker(sess)
    write_summary()
    print(json.dumps({"visible": visible, "no_token": no_token, "calls": str(CALLS), "summary": str(SUMMARY)}, indent=2), flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
