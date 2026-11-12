"""Container-only Pi-shaped transcript/control fixture for compaction UI checks.

No provider or backend CLI is involved. The final six native entries are copied
from the stripped real Pi fixture; earlier generated history exercises prepend.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from datetime import datetime, timedelta, timezone

home = Path.home()
app_dir = home / ".local/share/codoxear"
socks = app_dir / "socks"
socks.mkdir(parents=True, exist_ok=True)
log = home / "compaction.jsonl"
base = datetime(2026, 11, 12, 14, tzinfo=timezone.utc)


def message(index, role, text):
    return {
        "type": "message", "id": f"history-{index}",
        "timestamp": (base + timedelta(minutes=index)).isoformat(),
        "message": {"role": role, "content": [{"type": "text", "text": text}], **({"stopReason": "stop"} if role == "assistant" else {})},
    }


rows = [{"type": "session", "version": 3, "id": "compaction-thread", "cwd": "/workspace", "timestamp": base.isoformat()}]
for index in range(90):
    rows.append(message(index, "user" if index % 2 == 0 else "assistant", f"Earlier {'request' if index % 2 == 0 else 'answer'} {index}."))
    if index == 15:
        rows.append({"type": "compaction", "id": "older-compaction", "timestamp": (base + timedelta(minutes=index, seconds=1)).isoformat(), "tokensBefore": 90000, "summary": "NEVER DISPLAY OLDER SUMMARY"})
real = [json.loads(line) for line in (Path(__file__).parent / "compaction/pi-real.jsonl").read_text().splitlines()]
rows.extend(real)
# A tall current reply makes the divider reachable through normal scroll, and
# keeps user-neighbor navigation distinct from next-button assistant-top mode.
rows[-1]["message"]["content"] = [{"type": "text", "text": "Continuing after compaction.\n\n" + "\n\n".join(f"Paragraph {n}: the transcript preserves earlier conversation and resumes with a smaller working context." for n in range(1, 18))}]
log.write_text("".join(json.dumps(row) + "\n" for row in rows))
sock = socks / "compaction.sock"
sock.unlink(missing_ok=True)
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(str(sock))
server.listen(16)
pid = os.getpid()
sock.with_suffix(".json").write_text(json.dumps({
    "session_id": "compaction", "thread_id": "compaction-thread", "agent_backend": "pi", "owner": "web",
    "broker_pid": pid, "codex_pid": pid, "cwd": "/workspace", "start_ts": base.timestamp(),
    "log_path": str(log), "model_provider": "fixture", "model": "fixture-model",
    "control_protocol_version": 2, "control_capabilities": {"sync_send": True, "key_write_errors": True},
}))
while True:
    conn, _ = server.accept()
    with conn:
        data = b""
        while b"\n" not in data:
            block = conn.recv(65536)
            if not block:
                break
            data += block
        try:
            req = json.loads(data.split(b"\n", 1)[0])
            cmd = req.get("cmd")
            if cmd == "state":
                result = {"busy": False, "queue_len": 0, "token": None, "interrupted_idle": False}
            elif cmd == "tail":
                result = {"tail": ""}
            else:
                result = {"error": "read-only verification fixture"}
            conn.sendall(json.dumps(result).encode() + b"\n")
        except (ValueError, OSError):
            pass
