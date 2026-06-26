from __future__ import annotations

import os
import secrets
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NoReturn

from .agent_backend import get_agent_backend
from .backend_launch import build_tmux_inline_env
from .backend_launch import tmux_unset_vars
from .file_upload import safe_filename
from .launch_ledger import LaunchAttemptRecorder
from .sidecar_metadata import read_metadata as _read_sidecar_metadata
from .sidecar_metadata import required_live_pid as _metadata_required_live_pid


@dataclass(frozen=True)
class LaunchProcessRequest:
    argv: list[str]
    env: dict[str, str]
    agent_backend: str
    spawn_cwd: Path
    launch_id: str
    spawn_nonce: str
    create_in_tmux: bool
    tmux_session_name: str
    repo_root: Path
    resume_session_id: str | None = None
    model_provider: str | None = None
    preferred_auth_method: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None


@dataclass(frozen=True)
class LaunchProcessDeps:
    which_tmux: Callable[[str], str | None]
    run: Callable[..., subprocess.CompletedProcess[str]]
    popen: Callable[..., Any]
    wait_or_raise: Callable[..., None]
    wait_for_spawned_broker_meta: Callable[..., dict[str, Any]]
    tmux_pane_snapshot: Callable[..., dict[str, Any]]
    drain_stream: Callable[[Any], None]


@dataclass(frozen=True)
class LaunchContextRequest:
    argv: list[str]
    env: dict[str, str]
    agent_backend: str
    spawn_cwd: Path
    requested_cwd: str
    create_in_tmux: bool
    tmux_session_name: str
    repo_root: Path
    resume_session_id: str | None = None
    worktree_branch: str | None = None
    model_provider: str | None = None
    preferred_auth_method: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None


@dataclass(frozen=True)
class LaunchProcessContext:
    request: LaunchProcessRequest
    recorder: LaunchAttemptRecorder


class LaunchProcessFailure(RuntimeError):
    def __init__(self, record: dict[str, Any]):
        super().__init__(str(record.get("error") or record.get("stage") or "session launch failed"))
        self.record = record


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def prepare_launch_process_context(
    context: LaunchContextRequest,
    *,
    record_launch_attempt: Callable[[dict[str, Any]], dict[str, Any]],
    now: Callable[[], float] | None = None,
    token_hex: Callable[[int], str] | None = None,
    stderr: Any | None = None,
) -> LaunchProcessContext:
    now_fn = now or time.time
    token_fn = token_hex or secrets.token_hex
    launch_started_ts = now_fn()
    launch_id = f"launch-{int(launch_started_ts * 1000)}-{token_fn(4)}"
    spawn_nonce = token_fn(8)
    context.env["CODEX_WEB_LAUNCH_ID"] = launch_id
    context.env["CODEX_WEB_SPAWN_NONCE"] = spawn_nonce
    base_launch_record: dict[str, Any] = {
        "launch_id": launch_id,
        "state": "starting",
        "agent_backend": context.agent_backend,
        "cwd": str(context.spawn_cwd),
        "requested_cwd": context.requested_cwd,
        "created_ts": launch_started_ts,
        "updated_ts": launch_started_ts,
        "spawn_nonce": spawn_nonce,
        "model_provider": context.model_provider,
        "preferred_auth_method": context.preferred_auth_method,
        "model": context.model,
        "reasoning_effort": context.reasoning_effort,
        "service_tier": context.service_tier,
        "resume_session_id": context.resume_session_id,
        "worktree_branch": context.worktree_branch,
    }
    recorder = LaunchAttemptRecorder(
        base_launch_record,
        record_launch_attempt=record_launch_attempt,
        now=now_fn,
        stderr=stderr,
    )
    request = LaunchProcessRequest(
        argv=context.argv,
        env=context.env,
        agent_backend=context.agent_backend,
        spawn_cwd=context.spawn_cwd,
        launch_id=launch_id,
        spawn_nonce=spawn_nonce,
        create_in_tmux=context.create_in_tmux,
        tmux_session_name=context.tmux_session_name,
        repo_root=context.repo_root,
        resume_session_id=context.resume_session_id,
        model_provider=context.model_provider,
        preferred_auth_method=context.preferred_auth_method,
        model=context.model,
        reasoning_effort=context.reasoning_effort,
        service_tier=context.service_tier,
    )
    return LaunchProcessContext(request=request, recorder=recorder)


def wait_or_raise(
    proc: Any,
    *,
    label: str,
    timeout_s: float = 1.5,
    now: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    now_fn = now or time.time
    sleep_fn = sleep or time.sleep
    deadline = now_fn() + float(timeout_s)
    while now_fn() < deadline:
        rc = proc.poll()
        if rc is None:
            sleep_fn(0.05)
            continue
        _out, err = proc.communicate(timeout=0.5)
        err2 = err if isinstance(err, (bytes, bytearray)) else b""
        msg = bytes(err2).decode("utf-8", errors="replace").strip()
        msg = msg[-4000:] if msg else ""
        raise RuntimeError(f"{label} exited early (rc={rc}): {msg}")


def drain_stream(stream: Any) -> None:
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
    stream.close()


def wait_for_spawned_broker_meta(
    spawn_nonce: str,
    *,
    sock_dir: Path,
    timeout_s: float,
    now: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    read_metadata: Callable[..., dict[str, Any]] = _read_sidecar_metadata,
    required_live_pid: Callable[..., int] = _metadata_required_live_pid,
) -> dict[str, Any]:
    now_fn = now or time.time
    sleep_fn = sleep or time.sleep
    deadline = now_fn() + max(timeout_s, 0.0)
    while now_fn() <= deadline:
        for meta_path in sorted(sock_dir.glob("*.json")):
            sock = meta_path.with_suffix(".sock")
            try:
                meta = read_metadata(meta_path, sock=sock)
            except ValueError:
                continue
            if _clean_optional_text(meta.get("spawn_nonce")) != spawn_nonce:
                continue
            try:
                required_live_pid(meta, "broker_pid", sock=sock)
            except ValueError:
                continue
            return meta
        sleep_fn(0.05)
    raise RuntimeError(f"tmux launch did not publish broker metadata within {timeout_s:.1f}s")


def _tmux_detail(proc: subprocess.CompletedProcess[str]) -> str:
    return (proc.stderr or proc.stdout or f"exit status {proc.returncode}").strip()


def _tmux_missing_session(detail: str) -> bool:
    low = detail.lower()
    return "can't find session" in low or "no server running" in low or "error connecting to" in low


def _tmux_duplicate_session(detail: str) -> bool:
    return "duplicate session" in detail.lower()


def _tmux_launch_fields(snapshot: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    out = dict(snapshot or {})
    out.update(fields)
    return out


def launch_broker_process(
    request: LaunchProcessRequest,
    *,
    recorder: LaunchAttemptRecorder,
    deps: LaunchProcessDeps,
) -> dict[str, Any]:
    recorder.record("starting", transport="tmux" if request.create_in_tmux else "direct")

    def fail_launch(stage: str, error: BaseException | str, **extra: Any) -> NoReturn:
        raise LaunchProcessFailure(recorder.failure_record(stage, error, **extra))

    if request.create_in_tmux:
        return _launch_tmux_broker(request, recorder=recorder, deps=deps, fail_launch=fail_launch)
    return _launch_direct_broker(request, recorder=recorder, deps=deps, fail_launch=fail_launch)


def _launch_tmux_broker(
    request: LaunchProcessRequest,
    *,
    recorder: LaunchAttemptRecorder,
    deps: LaunchProcessDeps,
    fail_launch: Callable[..., NoReturn],
) -> dict[str, Any]:
    tmux_bin = deps.which_tmux("tmux")
    if tmux_bin is None:
        raise ValueError("tmux is unavailable on this host")

    tmux_window = safe_filename(f"{request.spawn_cwd.name or 'session'}-{request.spawn_nonce[:6]}", default="session")
    request.env["CODEX_WEB_TRANSPORT"] = "tmux"
    request.env["CODEX_WEB_TMUX_SESSION"] = request.tmux_session_name
    request.env["CODEX_WEB_TMUX_WINDOW"] = tmux_window
    backend_bin_env_var = get_agent_backend(request.agent_backend).bin_env_var
    inline_env = build_tmux_inline_env(
        request.env,
        agent_backend=request.agent_backend,
        tmux_session=request.tmux_session_name,
        tmux_window=tmux_window,
        launch_id=request.launch_id,
        spawn_nonce=request.spawn_nonce,
        resume_session_id=request.resume_session_id,
        model_provider=request.model_provider,
        preferred_auth_method=request.preferred_auth_method,
        model=request.model,
        reasoning_effort=request.reasoning_effort,
        service_tier=request.service_tier,
        inherited_backend_bin=_clean_optional_text(os.environ.get(backend_bin_env_var)),
    )
    inline_argv = ["env", *[f"{key}={value}" for key, value in inline_env.items()], *request.argv]
    shell_cmd = f"cd {shlex.quote(str(request.repo_root))} && unset {shlex.join(tmux_unset_vars())} && exec {shlex.join(inline_argv)}"
    new_window_argv = [
        tmux_bin,
        "new-window",
        "-d",
        "-P",
        "-F",
        "#{pane_id}",
        "-t",
        f"{request.tmux_session_name}:",
        "-n",
        tmux_window,
        shell_cmd,
    ]
    new_session_argv = [
        tmux_bin,
        "new-session",
        "-d",
        "-P",
        "-F",
        "#{pane_id}",
        "-s",
        request.tmux_session_name,
        "-n",
        tmux_window,
        shell_cmd,
    ]

    def tmux_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return deps.run(argv, capture_output=True, text=True, env=request.env, check=False)

    attempts: list[dict[str, Any]] = []
    tmux_proc = tmux_run(new_window_argv)
    attempts.append({"cmd": "new-window", "returncode": tmux_proc.returncode, "stderr": (tmux_proc.stderr or "").strip(), "stdout": (tmux_proc.stdout or "").strip()})
    if tmux_proc.returncode != 0 and _tmux_missing_session(_tmux_detail(tmux_proc)):
        tmux_proc = tmux_run(new_session_argv)
        attempts.append({"cmd": "new-session", "returncode": tmux_proc.returncode, "stderr": (tmux_proc.stderr or "").strip(), "stdout": (tmux_proc.stdout or "").strip()})
        if tmux_proc.returncode != 0 and _tmux_duplicate_session(_tmux_detail(tmux_proc)):
            tmux_proc = tmux_run(new_window_argv)
            attempts.append({"cmd": "new-window-after-duplicate", "returncode": tmux_proc.returncode, "stderr": (tmux_proc.stderr or "").strip(), "stdout": (tmux_proc.stdout or "").strip()})

    tmux_pane_id = _clean_optional_text(tmux_proc.stdout)
    if tmux_proc.returncode != 0:
        detail = _tmux_detail(tmux_proc)
        fail_launch(
            "tmux_launch",
            f"tmux launch failed: {detail}",
            transport="tmux",
            tmux_session=request.tmux_session_name,
            tmux_window=tmux_window,
            spawn_nonce=request.spawn_nonce,
            tmux_exit_status=tmux_proc.returncode,
            tmux_stdout=(tmux_proc.stdout or "").strip(),
            tmux_stderr=(tmux_proc.stderr or "").strip(),
            tmux_attempts=attempts,
        )

    snapshot = deps.tmux_pane_snapshot(tmux_bin, pane_id=tmux_pane_id, window=tmux_window)
    recorder.record(
        "tmux_pane_created",
        **_tmux_launch_fields(
            snapshot,
            transport="tmux",
            tmux_session=request.tmux_session_name,
            tmux_window=tmux_window,
            tmux_attempts=attempts,
        ),
    )
    try:
        meta = deps.wait_for_spawned_broker_meta(request.spawn_nonce)
    except Exception as e:
        if tmux_pane_id is not None and not snapshot.get("tmux_inspect_error") and str(snapshot.get("tmux_pane_dead") or "0") != "1":
            recorder.record(
                "tmux_pane_created",
                **_tmux_launch_fields(
                    snapshot,
                    stage="broker_metadata_pending",
                    error=str(e),
                    transport="tmux",
                    tmux_session=request.tmux_session_name,
                    tmux_window=tmux_window,
                ),
            )
            return {"launch_id": request.launch_id, "pending": True, "tmux_session": request.tmux_session_name, "tmux_window": tmux_window}
        fail_launch(
            "broker_metadata",
            e,
            **_tmux_launch_fields(
                deps.tmux_pane_snapshot(tmux_bin, pane_id=tmux_pane_id, window=tmux_window),
                transport="tmux",
                tmux_session=request.tmux_session_name,
                tmux_window=tmux_window,
                tmux_pane_id=tmux_pane_id,
                spawn_nonce=request.spawn_nonce,
            ),
        )

    broker_pid = meta.get("broker_pid")
    if not isinstance(broker_pid, int):
        fail_launch(
            "broker_metadata",
            "tmux launch metadata is missing broker_pid",
            transport="tmux",
            tmux_session=request.tmux_session_name,
            tmux_window=tmux_window,
            tmux_pane_id=tmux_pane_id,
            spawn_nonce=request.spawn_nonce,
            metadata=meta,
        )
    recorder.record(
        "broker_meta_bound",
        transport="tmux",
        tmux_session=request.tmux_session_name,
        tmux_window=tmux_window,
        tmux_pane_id=tmux_pane_id,
        broker_pid=int(broker_pid),
    )
    return {"broker_pid": int(broker_pid), "tmux_session": request.tmux_session_name, "tmux_window": tmux_window}


def _launch_direct_broker(
    request: LaunchProcessRequest,
    *,
    recorder: LaunchAttemptRecorder,
    deps: LaunchProcessDeps,
    fail_launch: Callable[..., NoReturn],
) -> dict[str, Any]:
    try:
        proc = deps.popen(
            request.argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=request.env,
            start_new_session=True,
        )
    except Exception as e:
        fail_launch("broker_spawn", f"spawn failed: {e}", transport="direct")

    try:
        deps.wait_or_raise(proc, label="broker", timeout_s=1.5)
    except Exception as e:
        fail_launch("broker_early_exit", e, transport="direct", broker_pid=int(proc.pid))
    recorder.record("broker_spawned", transport="direct", broker_pid=int(proc.pid))
    if proc.stderr is not None:
        threading.Thread(target=deps.drain_stream, args=(proc.stderr,), daemon=True).start()

    # Prevent zombies when the broker exits.
    threading.Thread(target=proc.wait, daemon=True).start()
    return {"broker_pid": int(proc.pid)}
