from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from codoxear.broker_turn_state import State
from codoxear.util import _paths_match
from codoxear.util import pid_alive as _pid_alive


def _claimed_log_paths_from_sock_meta(*, sock_dir: Path, exclude_sock: Path | None = None) -> set[Path]:
    out: set[Path] = set()
    if not sock_dir.exists():
        return out
    for meta_path in sock_dir.glob("*.json"):
        sock_path = meta_path.with_suffix(".sock")
        if exclude_sock is not None and _paths_match(sock_path, exclude_sock):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        log_path_raw = meta.get("log_path")
        if not isinstance(log_path_raw, str) or not log_path_raw.strip():
            continue
        broker_pid = int(meta.get("broker_pid")) if isinstance(meta.get("broker_pid"), int) else 0
        agent_pid = int(meta.get("codex_pid")) if isinstance(meta.get("codex_pid"), int) else 0
        if (broker_pid > 0 or agent_pid > 0) and (not _pid_alive(broker_pid)) and (not _pid_alive(agent_pid)):
            continue
        path = Path(log_path_raw)
        try:
            out.add(path.resolve())
        except Exception:
            out.add(path)
    return out


def _broker_sidecar_meta(
    st: State,
    *,
    owner_tag: str,
    agent_backend: str,
    model_provider: str,
    preferred_auth_method: str,
    model: str,
    reasoning_effort: str,
    service_tier: str,
) -> dict[str, Any]:
    return {
        "session_id": st.session_id,
        "owner": owner_tag if owner_tag else None,
        "broker_pid": os.getpid(),
        "sessiond_pid": os.getpid(),
        "codex_pid": st.codex_pid,
        "cwd": st.cwd,
        "start_ts": st.start_ts,
        "log_path": str(st.log_path) if st.log_path else None,
        "ignored_rollout_paths": sorted(str(p) for p in st.ignored_rollout_paths),
        "sock_path": str(st.sock_path),
        "agent_backend": agent_backend,
        "launch_id": (os.environ.get("CODEX_WEB_LAUNCH_ID") or "").strip() or None,
        "resume_session_id": st.resume_session_id,
        "model_provider": model_provider or None,
        "preferred_auth_method": preferred_auth_method or None,
        "model": model or None,
        "reasoning_effort": reasoning_effort or None,
        "service_tier": service_tier or None,
        "transport": (os.environ.get("CODEX_WEB_TRANSPORT") or "").strip() or None,
        "tmux_session": (os.environ.get("CODEX_WEB_TMUX_SESSION") or "").strip() or None,
        "tmux_window": (os.environ.get("CODEX_WEB_TMUX_WINDOW") or "").strip() or None,
        "spawn_nonce": (os.environ.get("CODEX_WEB_SPAWN_NONCE") or "").strip() or None,
        "control_protocol_version": 2,
        "control_capabilities": {"sync_send": True, "key_write_errors": True},
        "pi_thinking_command": bool(st.pi_thinking_command),
    }


def _write_broker_sidecar_meta(
    st: State,
    *,
    sock_dir: Path,
    owner_tag: str,
    agent_backend: str,
    model_provider: str,
    preferred_auth_method: str,
    model: str,
    reasoning_effort: str,
    service_tier: str,
) -> None:
    if not st.sock_path:
        return
    meta = _broker_sidecar_meta(
        st,
        owner_tag=owner_tag,
        agent_backend=agent_backend,
        model_provider=model_provider,
        preferred_auth_method=preferred_auth_method,
        model=model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
    )
    meta_path = st.sock_path.with_suffix(".json")
    sock_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    os.chmod(meta_path, 0o600)
