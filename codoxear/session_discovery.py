from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import sys

from .agent_backend import normalize_agent_backend
from .launch_config import clean_optional_text
from .launch_config import normalize_requested_service_tier
from .sidecar_metadata import ignored_rollout_paths as _metadata_ignored_rollout_paths
from .sidecar_metadata import key_write_errors_supported as _metadata_key_write_errors_supported
from .sidecar_metadata import log_invalid as _log_invalid_sidecar_metadata
from .sidecar_metadata import log_path as _metadata_log_path
from .sidecar_metadata import read_metadata as _read_sidecar_metadata
from .sidecar_metadata import required_int as _metadata_required_int
from .sidecar_metadata import required_text as _metadata_required_text
from .sidecar_metadata import start_ts as _metadata_start_ts
from .sidecar_metadata import sync_send_supported as _metadata_sync_send_supported


@dataclass(frozen=True)
class DiscoveryRegistration:
    session_id: str
    thread_id: str
    broker_pid: int
    codex_pid: int
    agent_backend: str
    owned: bool
    transport: str | None
    start_ts: float
    cwd: str
    log_path: Path | None
    sock_path: Path
    busy: bool
    queue_len: int
    token: dict[str, Any] | None
    meta_log_off: int
    model_provider: str | None
    preferred_auth_method: str | None
    model: str | None
    reasoning_effort: str | None
    service_tier: str | None
    tmux_session: str | None
    tmux_window: str | None
    launch_id: str | None
    spawn_nonce: str | None
    resume_session_id: str | None
    sync_send_supported: bool
    key_write_errors_supported: bool
    interrupted_idle: bool


@dataclass(frozen=True)
class DiscoveryStaleAction:
    session_id: str
    sock_path: Path
    meta_path: Path
    clear_session_state: bool = False
    unhide_session: bool = False
    failure_record: dict[str, Any] | None = None


@dataclass(frozen=True)
class DiscoveryRecentCwd:
    cwd: str
    ts: Any


@dataclass(frozen=True)
class DiscoveryResult:
    registrations: list[DiscoveryRegistration] = field(default_factory=list)
    stale_actions: list[DiscoveryStaleAction] = field(default_factory=list)
    recent_cwds: list[DiscoveryRecentCwd] = field(default_factory=list)


SessionTransportFunc = Callable[[dict[str, Any]], tuple[str | None, str | None, str | None]]
SessionRunSettingsFunc = Callable[[dict[str, Any], Path | None, str], tuple[str | None, str | None, str | None, str | None]]
ReadSessionMetaFunc = Callable[[Path, str, str], dict[str, Any] | None]
CoerceMainThreadLogFunc = Callable[[str, Path], tuple[str, Path]]
ProcFindOpenLogFunc = Callable[[Path, int, str, str, set[Path]], Path | None]
SockCallFunc = Callable[[Path, dict[str, Any], float], dict[str, Any]]
BrokerBusyQueueFunc = Callable[[dict[str, Any]], tuple[bool, int]]
BrokerInterruptedIdleFunc = Callable[[dict[str, Any]], bool]
SockErrorStaleFunc = Callable[[BaseException], bool]
PidAliveFunc = Callable[[int], bool]
TokenFinderFunc = Callable[[Path], dict[str, Any] | None]


@dataclass(frozen=True)
class DiscoveryDeps:
    pid_alive: PidAliveFunc
    proc_find_open_rollout_log: ProcFindOpenLogFunc
    read_session_meta_or_none: ReadSessionMetaFunc
    coerce_main_thread_log: CoerceMainThreadLogFunc
    session_transport: SessionTransportFunc
    session_run_settings: SessionRunSettingsFunc
    sock_call: SockCallFunc
    broker_busy_queue_from_state: BrokerBusyQueueFunc
    broker_interrupted_idle_from_state: BrokerInterruptedIdleFunc
    sock_error_definitely_stale: SockErrorStaleFunc
    token_update_finder: TokenFinderFunc


def discover_sessions(
    sock_dir: Path,
    *,
    proc_root: Path,
    hidden_sessions: set[str],
    deps: DiscoveryDeps,
) -> DiscoveryResult:
    sock_dir.mkdir(parents=True, exist_ok=True)
    registrations: list[DiscoveryRegistration] = []
    stale_actions: list[DiscoveryStaleAction] = []
    recent_cwds: list[DiscoveryRecentCwd] = []

    for sock in sorted(sock_dir.glob("*.sock")):
        session_id = sock.stem
        meta_path = sock.with_suffix(".json")
        if not meta_path.exists():
            stale_actions.append(
                DiscoveryStaleAction(
                    session_id=session_id,
                    sock_path=sock,
                    meta_path=meta_path,
                    clear_session_state=True,
                    unhide_session=True,
                )
            )
            continue

        try:
            meta = _read_sidecar_metadata(meta_path, sock=sock)
            codex_pid = _metadata_required_int(meta, "codex_pid", sock=sock)
            broker_pid = _metadata_required_int(meta, "broker_pid", sock=sock)
            cwd = _metadata_required_text(meta, "cwd", sock=sock)
            log_path = _metadata_log_path(meta, sock=sock)
            ignored_paths = _metadata_ignored_rollout_paths(meta, sock=sock)
            start_ts = _metadata_start_ts(meta, sock=sock)
            agent_backend = normalize_agent_backend(meta.get("agent_backend"), default="codex")
        except ValueError as e:
            _log_invalid_sidecar_metadata("discover", sock, e)
            continue

        thread_id = meta.get("session_id") if isinstance(meta.get("session_id"), str) and meta.get("session_id") else session_id
        owned = (meta.get("owner") == "web") if isinstance(meta.get("owner"), str) else False
        transport, tmux_session, tmux_window = deps.session_transport(meta)
        sync_send_supported = _metadata_sync_send_supported(meta)
        key_write_errors_supported = _metadata_key_write_errors_supported(meta)
        launch_id = clean_optional_text(meta.get("launch_id"))
        spawn_nonce = clean_optional_text(meta.get("spawn_nonce"))

        if log_path is not None and not log_path.exists():
            log_path = None
        if log_path is None and agent_backend in {"codex", "cc"} and deps.pid_alive(codex_pid):
            discovered_log_path = deps.proc_find_open_rollout_log(
                proc_root,
                codex_pid,
                agent_backend,
                cwd,
                ignored_paths,
            )
            if discovered_log_path is not None and discovered_log_path.exists():
                log_path = discovered_log_path
        if log_path is not None and agent_backend == "codex":
            session_meta = deps.read_session_meta_or_none(log_path, "codex", "session discovery")
            meta_session_id = session_meta.get("id") if session_meta else None
            if isinstance(meta_session_id, str) and meta_session_id:
                thread_id = meta_session_id
            thread_id, log_path = deps.coerce_main_thread_log(thread_id, log_path)

        if (log_path is None) and (not deps.pid_alive(codex_pid)) and (not deps.pid_alive(broker_pid)):
            failure_record = None
            if owned:
                failure_record = {
                    "launch_id": launch_id,
                    "state": "failed",
                    "stage": "broker_exit_before_log_bind",
                    "error": "broker exited before publishing a session log",
                    "agent_backend": agent_backend,
                    "cwd": meta.get("cwd"),
                    "created_ts": meta.get("start_ts"),
                    "broker_pid": broker_pid,
                    "agent_pid": codex_pid,
                    "transport": transport,
                    "tmux_session": tmux_session,
                    "tmux_window": tmux_window,
                    "spawn_nonce": spawn_nonce,
                    "model_provider": meta.get("model_provider"),
                    "preferred_auth_method": meta.get("preferred_auth_method"),
                    "model": meta.get("model"),
                    "reasoning_effort": meta.get("reasoning_effort"),
                    "service_tier": meta.get("service_tier"),
                }
            stale_actions.append(
                DiscoveryStaleAction(
                    session_id=session_id,
                    sock_path=sock,
                    meta_path=meta_path,
                    unhide_session=True,
                    failure_record=failure_record,
                )
            )
            continue

        if session_id in hidden_sessions:
            if (not deps.pid_alive(codex_pid)) and (not deps.pid_alive(broker_pid)):
                stale_actions.append(
                    DiscoveryStaleAction(
                        session_id=session_id,
                        sock_path=sock,
                        meta_path=meta_path,
                        unhide_session=True,
                    )
                )
            continue

        recent_cwds.append(DiscoveryRecentCwd(cwd=cwd, ts=meta.get("updated_ts", meta.get("start_ts"))))

        resume_session_id = clean_optional_text(meta.get("resume_session_id"))
        model_provider, preferred_auth_method, model, reasoning_effort = deps.session_run_settings(meta, log_path, agent_backend)
        service_tier = normalize_requested_service_tier(meta.get("service_tier")) if agent_backend == "codex" else None

        try:
            resp = deps.sock_call(sock, {"cmd": "state"}, 0.5)
        except Exception as e:
            sys.stderr.write(f"error: discover: sock state call failed for {sock}: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
            if deps.sock_error_definitely_stale(e) and (not deps.pid_alive(codex_pid)) and (not deps.pid_alive(broker_pid)):
                stale_actions.append(
                    DiscoveryStaleAction(
                        session_id=session_id,
                        sock_path=sock,
                        meta_path=meta_path,
                    )
                )
            continue

        if log_path is not None:
            meta_log_off = int(log_path.stat().st_size)
            token = deps.token_update_finder(log_path)
        else:
            meta_log_off = 0
            token = None
        try:
            broker_busy, broker_queue_len = deps.broker_busy_queue_from_state(resp)
            broker_interrupted_idle = deps.broker_interrupted_idle_from_state(resp)
        except ValueError as e:
            sys.stderr.write(f"error: discover: invalid broker state for {sock}: {e}\n")
            sys.stderr.flush()
            continue
        if token is None and log_path is None:
            token = resp.get("token") if isinstance(resp.get("token"), (dict, type(None))) else None

        registrations.append(
            DiscoveryRegistration(
                session_id=session_id,
                thread_id=thread_id,
                broker_pid=broker_pid,
                codex_pid=codex_pid,
                agent_backend=agent_backend,
                owned=owned,
                transport=transport,
                start_ts=float(start_ts),
                cwd=str(cwd),
                log_path=log_path,
                sock_path=sock,
                busy=broker_busy,
                queue_len=broker_queue_len,
                token=token,
                meta_log_off=meta_log_off,
                model_provider=model_provider,
                preferred_auth_method=preferred_auth_method,
                model=model,
                reasoning_effort=reasoning_effort,
                service_tier=service_tier,
                tmux_session=tmux_session,
                tmux_window=tmux_window,
                launch_id=launch_id,
                spawn_nonce=spawn_nonce,
                resume_session_id=resume_session_id,
                sync_send_supported=sync_send_supported,
                key_write_errors_supported=key_write_errors_supported,
                interrupted_idle=broker_interrupted_idle,
            )
        )

    return DiscoveryResult(registrations=registrations, stale_actions=stale_actions, recent_cwds=recent_cwds)
