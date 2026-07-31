from __future__ import annotations
import json, time
from pathlib import Path
from codoxear.util import append_launch_attempt, launch_attempts_path
now = time.time()
records = []
for i, (launch_id, cwd_name) in enumerate([
    ("launch-confirm-cancel", "confirm-cancel-project"),
    ("launch-confirm-delete", "confirm-delete-project"),
]):
    rec = {
        "type": "launch_attempt",
        "launch_id": launch_id,
        "state": "failed",
        "stage": "in_app_confirm_browser_proof",
        "error": f"Synthetic failed launch for in-app confirm browser proof: {launch_id}",
        "agent_backend": "codex",
        "broker_pid": 999999,
        "created_ts": now + i,
        "updated_ts": now + i,
        "cwd": f"/workspace/{cwd_name}",
        "requested_cwd": f"/workspace/{cwd_name}",
        "model": "gpt-confirm-proof",
        "model_provider": "openai-api",
        "provider_choice": "openai-api",
        "reasoning_effort": "high",
        "transport": "direct",
        "spawn_nonce": f"confirm-proof-{i}",
    }
    append_launch_attempt(rec, path=launch_attempts_path())
    records.append(rec)
print(json.dumps({"path": str(launch_attempts_path()), "records": records}, indent=2))
