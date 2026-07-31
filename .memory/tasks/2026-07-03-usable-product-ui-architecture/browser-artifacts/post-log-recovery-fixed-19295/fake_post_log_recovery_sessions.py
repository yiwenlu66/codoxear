#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
from pathlib import Path

APP = Path(os.environ.get("HOME", "/home/tester")) / ".local" / "share" / "codoxear"
SOCKS = APP / "socks"
LOGS = APP / "post-log-recovery-proof-logs"
LEDGER = APP / "session_launches.jsonl"
POST_LOG_RECOVERY_TRANSCRIPT_MAX_BYTES = 2 * 1024 * 1024
SENTINEL = "POST_LOG_BOUND_DEATH_SENTINEL"
STOPPED = "The backend process stopped before completing this turn."
RECOVERY_SID = "post-log-recovery-fixed"
RECOVERY_THREAD = "thread-post-log-recovery-fixed"
RECOVERY_LAUNCH = "launch-post-log-recovery-fixed"
CONTROL_SID = "post-log-completed-control"
CONTROL_THREAD = "thread-post-log-completed-control"
CONTROL_LAUNCH = "launch-post-log-completed-control"
LARGE_SID = "post-log-large-cursor"
LARGE_THREAD = "thread-post-log-large-cursor"
LARGE_LAUNCH = "launch-post-log-large-cursor"
FIRST = "FIRST_EVENT_SENTINEL"
DEAD_BROKER_PID = 987654321
DEAD_AGENT_PID = 987654322
START_TS = 1783312000.0


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def pi_session_row(thread: str) -> dict:
    return {"type": "session", "id": thread, "cwd": "/workspace", "timestamp": "2026-07-06T00:00:00Z"}


def pi_user_row(text: str) -> dict:
    return {
        "type": "message",
        "ts": 1.0,
        "timestamp": "2026-07-06T00:00:01Z",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def pi_assistant_row(text: str) -> dict:
    return {
        "type": "message",
        "ts": 2.0,
        "timestamp": "2026-07-06T00:00:02Z",
        "message": {"role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": text}]},
    }


def write_large_codex_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = {"type": "event_msg", "ts": 1.0, "payload": {"type": "user_message", "message": FIRST}}
    filler = {"type": "debug", "payload": "x" * 4096}
    filler_line = json.dumps(filler, separators=(",", ":")) + "\n"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(first, separators=(",", ":")) + "\n")
        while f.tell() <= POST_LOG_RECOVERY_TRANSCRIPT_MAX_BYTES + 4096:
            f.write(filler_line)


def stale_socket(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    s.close()


def write_sidecar(sid: str, thread: str, launch: str, log_path: Path, *, backend: str) -> None:
    meta = {
        "session_id": thread,
        "agent_backend": backend,
        "owner": "web",
        "broker_pid": DEAD_BROKER_PID,
        "codex_pid": DEAD_AGENT_PID,
        "cwd": "/workspace",
        "start_ts": START_TS,
        "updated_ts": START_TS + 5,
        "log_path": str(log_path),
        "launch_id": launch,
        "spawn_nonce": f"spawn-{sid}",
        "model_provider": "proof-provider",
        "preferred_auth_method": "apikey",
        "model": "proof-model",
        "reasoning_effort": "low",
        "control_protocol_version": 2,
        "control_capabilities": {"sync_send": True, "key_write_errors": True},
    }
    (SOCKS / f"{sid}.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n", encoding="utf-8")


def append_launch_state(launch: str, sid: str, thread: str, log_path: Path, *, backend: str) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "type": "launch_attempt",
        "launch_id": launch,
        "session_id": sid,
        "thread_id": thread,
        "state": "log_bound",
        "agent_backend": backend,
        "cwd": "/workspace",
        "created_ts": START_TS,
        "updated_ts": START_TS + 1,
        "log_path": str(log_path),
        "broker_pid": DEAD_BROKER_PID,
        "agent_pid": DEAD_AGENT_PID,
        "spawn_nonce": f"spawn-{sid}",
        "model_provider": "proof-provider",
        "preferred_auth_method": "apikey",
        "model": "proof-model",
        "reasoning_effort": "low",
    }
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


SOCKS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
recovery_log = LOGS / f"{RECOVERY_SID}.jsonl"
control_log = LOGS / f"{CONTROL_SID}.jsonl"
large_log = LOGS / f"{LARGE_SID}.jsonl"
write_jsonl(recovery_log, [pi_session_row(RECOVERY_THREAD), pi_user_row(SENTINEL)])
write_jsonl(control_log, [pi_session_row(CONTROL_THREAD), pi_user_row("completed control prompt"), pi_assistant_row("completed control final answer")])
write_large_codex_log(large_log)
for sid, thread, launch, log, backend in [
    (RECOVERY_SID, RECOVERY_THREAD, RECOVERY_LAUNCH, recovery_log, "pi"),
    (CONTROL_SID, CONTROL_THREAD, CONTROL_LAUNCH, control_log, "pi"),
    (LARGE_SID, LARGE_THREAD, LARGE_LAUNCH, large_log, "codex"),
]:
    stale_socket(SOCKS / f"{sid}.sock")
    write_sidecar(sid, thread, launch, log, backend=backend)
    append_launch_state(launch, sid, thread, log, backend=backend)

print(json.dumps({
    "ready": True,
    "app": str(APP),
    "socks": sorted(p.name for p in SOCKS.glob("post-log-*.sock")),
    "sidecars": sorted(p.name for p in SOCKS.glob("post-log-*.json")),
    "logs": {"recovery": str(recovery_log), "control": str(control_log), "large": str(large_log), "large_size": large_log.stat().st_size},
    "ledger": str(LEDGER),
    "sentinel": SENTINEL,
    "stopped": STOPPED,
    "first": FIRST,
}, indent=2, sort_keys=True))
