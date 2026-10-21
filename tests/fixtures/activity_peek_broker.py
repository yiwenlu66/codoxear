"""Container-only broker fixture for activity peek verification.

Registers one fake Pi session whose log holds an open turn with an in-flight
`bash: sleep 30` tool call. The verify script appends rows to the log through
docker exec to advance the scenario (result clears the peek, a second call
replaces it, a final text closes the turn). Busy answers come from the control
socket and flip only when the idle marker file appears.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket


home = Path.home()
app_dir = home / ".local" / "share" / "codoxear"
socks_dir = app_dir / "socks"
socks_dir.mkdir(parents=True, exist_ok=True)
sock_path = socks_dir / "activity-peek.sock"
meta_path = sock_path.with_suffix(".json")
log_path = home / "activity-peek.jsonl"
idle_path = home / "activity-peek-idle"
now = "2026-10-21T06:00:00.000Z"
rows = [
    {"type": "session", "version": 3, "id": "activity-peek-thread", "cwd": "/workspace", "timestamp": now},
    {
        "type": "message",
        "id": "u1",
        "parentId": None,
        "timestamp": now,
        "message": {"role": "user", "content": [{"type": "text", "text": "Run a slow command."}]},
    },
    {
        "type": "message",
        "id": "a1",
        "parentId": "u1",
        "timestamp": now,
        "message": {
            "role": "assistant",
            "model": "provider/parent-model",
            "stopReason": "toolUse",
            "content": [
                {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "sleep 30"}},
            ],
        },
    },
]
log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

try:
    sock_path.unlink()
except FileNotFoundError:
    pass
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(str(sock_path))
server.listen(16)
pid = os.getpid()
meta = {
    "session_id": "activity-peek",
    "agent_backend": "pi",
    "owner": "web",
    "broker_pid": pid,
    "codex_pid": pid,
    "cwd": "/workspace",
    "start_ts": 1789103548.0,
    "updated_ts": 1789103548.0,
    "log_path": str(log_path),
    "model_provider": "provider",
    "model": "parent-model",
    "reasoning_effort": "high",
    "control_protocol_version": 2,
    "control_capabilities": {"sync_send": True, "key_write_errors": True},
}
meta_path.write_text(json.dumps(meta), encoding="utf-8")

while True:
    connection, _ = server.accept()
    with connection:
        data = b""
        while not data.endswith(b"\n"):
            chunk = connection.recv(65536)
            if not chunk:
                break
            data += chunk
        try:
            request = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, TypeError):
            request = {}
        if request.get("cmd") == "state":
            busy = not idle_path.exists()
            response = {"busy": busy, "turn_open": busy, "queue_len": 0, "interrupted_idle": False, "token": None}
        else:
            response = {"error": "unknown cmd"}
        connection.sendall(json.dumps(response).encode("utf-8") + b"\n")
