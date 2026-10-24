"""Container-only broker fixture for subagent activity layout verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket


home = Path.home()
app_dir = home / ".local" / "share" / "codoxear"
socks_dir = app_dir / "socks"
socks_dir.mkdir(parents=True, exist_ok=True)
sock_path = socks_dir / "subagent-layout.sock"
meta_path = sock_path.with_suffix(".json")
log_path = home / "subagent-layout.jsonl"
idle_path = home / "subagent-layout-idle"
models_path = home / "subagent-layout-models.json"
now = "2026-09-11T06:00:00.000Z"
rows = [
    {"type": "session", "version": 3, "id": "subagent-layout-thread", "cwd": "/workspace", "timestamp": now},
    {
        "type": "message",
        "id": "u1",
        "parentId": None,
        "timestamp": now,
        "message": {"role": "user", "content": [{"type": "text", "text": "Run two child tasks in parallel."}]},
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
            "usage": {"reasoning": 5500},
            "content": [
                {"type": "thinking", "thinking": "Coordinating two child tasks."},
                {"type": "toolCall", "id": "tool-1", "name": "read", "arguments": {"path": "one"}},
                {"type": "toolCall", "id": "tool-2", "name": "read", "arguments": {"path": "two"}},
            ],
        },
    },
]
log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
models_path.write_text(json.dumps(["dexgem-responses/gpt-5.6-sol", "dexgem-responses/gpt-5.3"]), encoding="utf-8")

try:
    sock_path.unlink()
except FileNotFoundError:
    pass
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(str(sock_path))
server.listen(16)
pid = os.getpid()
meta = {
    "session_id": "subagent-layout",
    "agent_backend": "pi",
    "owner": "web",
    "broker_pid": pid,
    "codex_pid": pid,
    "cwd": "/workspace",
    "start_ts": 1789103548.0,
    "updated_ts": 1789103548.0,
    "log_path": str(log_path),
    "model_provider": "dexgem-responses",
    "model": "parent-model",
    "reasoning_effort": "high",
    "control_protocol_version": 2,
    "control_capabilities": {"sync_send": True, "key_write_errors": True},
}
meta_path.write_text(json.dumps(meta), encoding="utf-8")

status_root = Path(os.environ["CODEX_WEB_SUBAGENT_RUNS_ROOT"])
run_dir = status_root / "parallel-layout"
run_dir.mkdir(parents=True, exist_ok=True)
status_path = run_dir / "status.json"

# A child launched on a jev-auto placeholder routes itself after launch and records
# the real model in its OWN session file (pi-jev-auto-model's custom entry plus the
# model_change it applies). The parent's status.json keeps the placeholder, and the
# run's events.jsonl is not guaranteed to mirror the routing, so this fixture writes
# only the child session - the record a real routed child always has.
JEV_CHILD_MODELS = [
    ("dexgem-messages", "kimi-k3"),
    ("dexgem-completions", "deepseek-flash"),
]


def jev_child_session(index: int, placeholder: str) -> str | None:
    bare = placeholder.strip().split("/")[-1]
    name, _, level = bare.partition(":")
    if name != "auto-jev-subagent":
        return None
    provider, model_id = JEV_CHILD_MODELS[index]
    child_path = run_dir / f"child-{index}" / "run-0" / "session.jsonl"
    child_path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {"type": "session", "version": 3, "id": f"child-{index}", "cwd": "/workspace", "timestamp": now},
        {"type": "model_change", "provider": "jev-auto", "modelId": "auto-jev-subagent"},
        {"type": "thinking_level_change", "thinkingLevel": level},
        {"type": "model_change", "provider": provider, "modelId": model_id},
        {"type": "custom", "customType": "jev-auto-model:routed:v1", "data": {
            "placeholder": "jev-auto/auto-jev-subagent", "provider": provider, "modelId": model_id, "level": level,
        }},
    ]
    child_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return str(child_path)


def step_model_field(index: int, model: str) -> dict:
    """Real status.json omits sessionFile for children launched on a real model."""
    child = jev_child_session(index, model)
    return {"sessionFile": child} if child else {}


def refresh_status() -> None:
    try:
        models = json.loads(models_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        models = []
    if not isinstance(models, list) or len(models) != 2 or not all(isinstance(value, str) and value for value in models):
        models = ["dexgem-responses/gpt-5.6-sol", "dexgem-responses/gpt-5.3"]
    status = {
        "lifecycleArtifactVersion": 3,
        "runId": "parallel-layout",
        "sessionId": str(log_path),
        "mode": "parallel",
        "state": "running",
        "startedAt": 1789103548000,
        "pid": pid,
        "currentStep": 1,
        "parallelGroups": [{"start": 0, "count": 2, "stepIndex": 0}],
        "steps": [
            {
                "agent": "reviewer",
                "status": "running",
                "startedAt": 1789103548001,
                "model": models[0],
                **step_model_field(0, models[0]),
                "toolCount": 19,
                "tokens": {"input": 31000, "output": 3500, "total": 34500},
            },
            {
                "agent": "executor",
                "status": "running",
                "startedAt": 1789103548002,
                "model": models[1],
                **step_model_field(1, models[1]),
                "toolCount": 18,
                "tokens": {"input": 15000, "output": 2200, "total": 17200},
            },
        ],
    }
    status_path.write_text(json.dumps(status), encoding="utf-8")


refresh_status()
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
        refresh_status()
        if request.get("cmd") == "state":
            busy = not idle_path.exists()
            response = {"busy": busy, "turn_open": busy, "queue_len": 0, "interrupted_idle": False, "token": None}
        else:
            response = {"error": "unknown cmd"}
        connection.sendall(json.dumps(response).encode("utf-8") + b"\n")
