from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent_backend import normalize_agent_backend
from .runtime_access import manager_runtime


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def _parse_historical_session_id(session_id: str) -> tuple[str, str] | None:
    raw = str(session_id or "").strip()
    if not raw.startswith("history:"):
        return None
    _prefix, backend, resume_session_id = (
        raw.split(":", 2) if raw.count(":") >= 2 else ("", "", "")
    )
    backend_clean = normalize_agent_backend(backend, default="codex")
    resume_clean = _clean_optional_text(resume_session_id)
    if not resume_clean:
        return None
    return backend_clean, resume_clean


def _historical_session_row(manager: Any, session_id: str) -> dict[str, Any] | None:
    parsed = _parse_historical_session_id(session_id)
    if parsed is None:
        return None
    backend, resume_session_id = parsed
    for row in manager.list_sessions():
        if normalize_agent_backend(row.get("agent_backend", row.get("backend")), default="codex") != backend:
            continue
        if _clean_optional_text(row.get("resume_session_id")) != resume_session_id:
            continue
        out = dict(row)
        out["session_id"] = session_id
        out["resume_session_id"] = resume_session_id
        out["historical"] = True
        return out
    return None


def _listed_session_row(manager: Any, session_id: str) -> dict[str, Any] | None:
    for row in manager.list_sessions():
        if str(row.get("session_id") or "") == session_id:
            return dict(row)
    return None


def _resume_historical_pi_session(manager: Any, session_id: str) -> dict[str, Any] | None:
    historical_row = _historical_session_row(manager, session_id)
    if historical_row is None and manager._runtime_session_id_for_identifier(session_id) is None:
        listed_row = _listed_session_row(manager, session_id)
        if isinstance(listed_row, dict) and listed_row.get("historical"):
            historical_row = listed_row
    if historical_row is None:
        return None

    backend = normalize_agent_backend(
        historical_row.get("agent_backend", historical_row.get("backend")),
        default="codex",
    )
    if backend != "pi":
        raise KeyError("unknown session")
    cwd = _clean_optional_text(historical_row.get("cwd"))
    resume_session_id = _clean_optional_text(historical_row.get("resume_session_id"))
    if cwd is None or resume_session_id is None:
        raise ValueError("historical session is missing resume metadata")
    spawn_res = manager.spawn_web_session(
        cwd=cwd,
        backend="pi",
        resume_session_id=resume_session_id,
    )
    manager._discover_existing(force=True, skip_invalid_sidecars=True)
    live_runtime_id = _clean_optional_text(spawn_res.get("runtime_id"))
    live_session_id = _clean_optional_text(spawn_res.get("session_id"))
    if live_runtime_id is None or live_session_id is None:
        raise RuntimeError("spawned session did not return session identities")
    if manager._runtime_session_id_for_identifier(live_runtime_id) is None:
        raise RuntimeError("spawned session is not yet discoverable")
    return {
        "runtime_id": live_runtime_id,
        "session_id": live_session_id,
        "backend": "pi",
    }


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


@dataclass(slots=True)
class SessionControlService:
    manager: Any

    def send(self, session_id: str, text: str) -> dict[str, Any]:
        return send(self.manager, session_id, text)

    def enqueue(self, session_id: str, text: str) -> dict[str, Any]:
        return enqueue(self.manager, session_id, text)

    def queue_list(self, session_id: str) -> list[str]:
        return queue_list(self.manager, session_id)

    def queue_delete(self, session_id: str, index: int) -> dict[str, Any]:
        return queue_delete(self.manager, session_id, int(index))

    def queue_update(self, session_id: str, index: int, text: str) -> dict[str, Any]:
        return queue_update(self.manager, session_id, int(index), text)

    def spawn_web_session(
        self,
        *,
        cwd: str,
        args: list[str] | None = None,
        agent_backend: str = "codex",
        resume_session_id: str | None = None,
        worktree_branch: str | None = None,
        model_provider: str | None = None,
        preferred_auth_method: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        create_in_tmux: bool = False,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return spawn_web_session(
            self.manager,
            cwd=cwd,
            args=args,
            agent_backend=agent_backend,
            resume_session_id=resume_session_id,
            worktree_branch=worktree_branch,
            model_provider=model_provider,
            preferred_auth_method=preferred_auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            create_in_tmux=create_in_tmux,
            backend=backend,
        )

    def restart_session(self, session_id: str) -> dict[str, Any]:
        return restart_session(self.manager, session_id)

    def handoff_session(self, session_id: str) -> dict[str, Any]:
        return handoff_session(self.manager, session_id)


def service(manager: Any) -> SessionControlService:
    return SessionControlService(manager)


def send(manager: Any, session_id: str, text: str) -> dict[str, Any]:
    resumed = _resume_historical_pi_session(manager, session_id)
    if resumed is not None:
        resp = send(manager, resumed["runtime_id"], text)
        out = dict(resp)
        out["session_id"] = resumed["session_id"]
        out["runtime_id"] = resumed["runtime_id"]
        out["backend"] = resumed["backend"]
        return out

    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    with manager._lock:
        session = manager._sessions.get(runtime_id)
        if not session:
            raise KeyError("unknown session")
        durable_session_id = manager._durable_session_id_for_session(session)
    transport_state, transport_error = manager._probe_bridge_transport(runtime_id)
    if transport_state == "dead":
        with manager._lock:
            manager._sessions.pop(runtime_id, None)
        manager._clear_deleted_session_state(runtime_id)
        _unlink_quiet(session.sock_path)
        _unlink_quiet(session.sock_path.with_suffix(".json"))
        raise KeyError("unknown session")
    request = manager._enqueue_outbound_request(runtime_id, text)
    return {
        "ok": True,
        "accepted": True,
        "request_id": request.request_id,
        "delivery_state": request.state,
        "session_id": durable_session_id,
        "runtime_id": runtime_id,
        "backend": session.backend,
        "transport_state": transport_state,
        "transport_error": transport_error,
    }


def enqueue(manager: Any, session_id: str, text: str) -> dict[str, Any]:
    resumed = _resume_historical_pi_session(manager, session_id)
    if resumed is not None:
        resp = enqueue(manager, resumed["runtime_id"], text)
        out = dict(resp)
        out["session_id"] = resumed["session_id"]
        out["runtime_id"] = resumed["runtime_id"]
        out["backend"] = resumed["backend"]
        return out
    return manager._queue_enqueue_local(session_id, text)


def queue_list(manager: Any, session_id: str) -> list[str]:
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        raise KeyError("unknown session")
    return manager._queue_list_local(runtime_id)


def queue_delete(manager: Any, session_id: str, index: int) -> dict[str, Any]:
    return manager._queue_delete_local(session_id, int(index))


def queue_update(manager: Any, session_id: str, index: int, text: str) -> dict[str, Any]:
    return manager._queue_update_local(session_id, int(index), text)


def spawn_web_session(
    manager: Any,
    *,
    cwd: str,
    args: list[str] | None = None,
    agent_backend: str = "codex",
    resume_session_id: str | None = None,
    worktree_branch: str | None = None,
    model_provider: str | None = None,
    preferred_auth_method: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    create_in_tmux: bool = False,
    backend: str | None = None,
) -> dict[str, Any]:
    sv = manager_runtime(manager)

    backend_name = normalize_agent_backend(
        backend, default=normalize_agent_backend(agent_backend, default="codex")
    )
    cwd_path = sv._resolve_dir_target(cwd, field_name="cwd")
    if not cwd_path.exists():
        try:
            cwd_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            detail = exc.strerror or str(exc)
            raise ValueError(f"cwd could not be created: {cwd_path}: {detail}") from exc
    if not cwd_path.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd_path}")
    cwd3 = str(cwd_path)
    if backend_name == "pi":
        spawn_nonce = sv.secrets.token_hex(8)
        pending_session_id: str | None = None
        pending_delete_on_failure = True
        pending_restore_record: Any | None = None
        if resume_session_id is not None:
            resume_id = sv._clean_optional_resume_session_id(resume_session_id)
            if not resume_id:
                raise ValueError("resume_session_id must be a non-empty string")
            session_path: Path | None = None
            for row in sv._list_resume_candidates_for_cwd(cwd3, limit=1000, backend="pi"):
                if row.get("session_id") != resume_id:
                    continue
                raw_session_path = row.get("session_path")
                if isinstance(raw_session_path, str) and raw_session_path:
                    session_path = Path(raw_session_path)
                    break
            if session_path is None:
                raise ValueError(f"resume session not found for cwd: {resume_id}")
            if create_in_tmux:
                pending_session_id = resume_id
                pending_delete_on_failure = False
                db = getattr(manager, "_page_state_db", None)
                pending_restore_record = (
                    db.load_sessions().get(("pi", resume_id)) if isinstance(db, sv.PageStateDB) else None
                )
                current = pending_restore_record
                manager._persist_durable_session_record(
                    sv.DurableSessionRecord(
                        backend="pi",
                        session_id=resume_id,
                        cwd=(current.cwd if current is not None else cwd3),
                        source_path=(current.source_path if current is not None else str(session_path)),
                        title=current.title if current is not None else None,
                        first_user_message=current.first_user_message if current is not None else None,
                        created_at=(current.created_at if current is not None else sv._safe_path_mtime(session_path)),
                        updated_at=(current.updated_at if current is not None else sv._safe_path_mtime(session_path)),
                        pending_startup=True,
                    )
                )
        else:
            pending_session_id = str(sv.uuid.uuid4())
            session_path = sv._pi_new_session_file_for_cwd(cwd_path)
            sv._write_pi_session_header(
                session_path,
                session_id=pending_session_id,
                cwd=cwd3,
                provider=model_provider,
                model_id=model,
                thinking_level=reasoning_effort,
            )
            manager._persist_durable_session_record(
                sv.DurableSessionRecord(
                    backend="pi",
                    session_id=pending_session_id,
                    cwd=cwd3,
                    source_path=str(session_path),
                    created_at=sv._safe_path_mtime(session_path),
                    updated_at=sv._safe_path_mtime(session_path),
                    pending_startup=True,
                )
            )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            sv.sys.executable,
            "-m",
            "codoxear.pi_broker",
            "--cwd",
            str(cwd_path),
            "--session-file",
            str(session_path),
            "--",
            "-e",
            str(Path(sv.__file__).resolve().parent / "pi_extensions" / "ask_user_bridge.ts"),
        ]
        env = dict(sv.os.environ)
        if sv._DOTENV.exists():
            for key, value in sv._load_env_file(sv._DOTENV).items():
                env.setdefault(key, value)
        env["CODEX_WEB_OWNER"] = "web"
        env["CODEX_WEB_SPAWN_NONCE"] = spawn_nonce
        env.setdefault("PI_HOME", str(sv.PI_HOME))
        if create_in_tmux:
            tmux_bin = sv.shutil.which("tmux")
            if tmux_bin is None:
                if pending_session_id is not None and pending_delete_on_failure:
                    manager._delete_durable_session_record(("pi", pending_session_id))
                elif pending_restore_record is not None:
                    manager._persist_durable_session_record(pending_restore_record)
                raise ValueError("tmux is unavailable on this host")
            tmux_window = sv._safe_filename(f"{Path(cwd3).name or 'session'}-{spawn_nonce[:6]}", default="session")
            env["CODEX_WEB_TRANSPORT"] = "tmux"
            env["CODEX_WEB_TMUX_SESSION"] = sv.TMUX_SESSION_NAME
            env["CODEX_WEB_TMUX_WINDOW"] = tmux_window
            short_app_dir = sv._ensure_tmux_short_app_dir()
            inline_env = {
                "CODEX_WEB_OWNER": "web",
                "CODEX_WEB_AGENT_BACKEND": "pi",
                "CODEX_WEB_TRANSPORT": "tmux",
                "CODEX_WEB_TMUX_SESSION": sv.TMUX_SESSION_NAME,
                "CODEX_WEB_TMUX_WINDOW": tmux_window,
                "CODEX_WEB_SPAWN_NONCE": spawn_nonce,
                "CODOXEAR_APP_DIR": short_app_dir,
                "PI_HOME": str(env["PI_HOME"]),
            }
            repo_root = Path(sv.__file__).resolve().parent.parent
            inline_argv = ["env", *[f"{key}={value}" for key, value in inline_env.items()], *argv]
            shell_cmd = f"cd {sv.shlex.quote(str(repo_root))} && exec {sv.shlex.join(inline_argv)}"
            has_session = sv.subprocess.run(
                [tmux_bin, "has-session", "-t", sv.TMUX_SESSION_NAME],
                stdout=sv.subprocess.DEVNULL,
                stderr=sv.subprocess.DEVNULL,
                text=True,
                check=False,
            )
            if has_session.returncode == 0:
                tmux_argv = [
                    tmux_bin,
                    "new-window",
                    "-d",
                    "-P",
                    "-F",
                    "#{pane_id}",
                    "-t",
                    f"{sv.TMUX_SESSION_NAME}:",
                    "-n",
                    tmux_window,
                    shell_cmd,
                ]
            else:
                tmux_argv = [
                    tmux_bin,
                    "new-session",
                    "-d",
                    "-P",
                    "-F",
                    "#{pane_id}",
                    "-s",
                    sv.TMUX_SESSION_NAME,
                    "-n",
                    tmux_window,
                    shell_cmd,
                ]
            tmux_proc = sv.subprocess.run(tmux_argv, capture_output=True, text=True, env=env, check=False)
            if tmux_proc.returncode != 0:
                if pending_session_id is not None and pending_delete_on_failure:
                    manager._delete_durable_session_record(("pi", pending_session_id))
                elif pending_restore_record is not None:
                    manager._persist_durable_session_record(pending_restore_record)
                detail = (tmux_proc.stderr or tmux_proc.stdout or f"exit status {tmux_proc.returncode}").strip()
                raise RuntimeError(f"tmux launch failed: {detail}")
            if pending_session_id is not None:
                sv.threading.Thread(
                    target=manager._finalize_pending_pi_spawn,
                    kwargs={
                        "spawn_nonce": spawn_nonce,
                        "durable_session_id": pending_session_id,
                        "cwd": cwd3,
                        "session_path": session_path,
                        "proc": None,
                        "delete_on_failure": pending_delete_on_failure,
                        "restore_record_on_failure": pending_restore_record,
                    },
                    daemon=True,
                ).start()
                return {
                    "session_id": pending_session_id,
                    "runtime_id": None,
                    "backend": "pi",
                    "pending_startup": True,
                    "tmux_session": sv.TMUX_SESSION_NAME,
                    "tmux_window": tmux_window,
                }
            meta = sv._wait_for_spawned_broker_meta(spawn_nonce)
            payload = sv._spawn_result_from_meta(meta)
            return {**payload, "tmux_session": sv.TMUX_SESSION_NAME, "tmux_window": tmux_window}
        try:
            proc = sv.subprocess.Popen(
                argv,
                stdin=sv.subprocess.DEVNULL,
                stdout=sv.subprocess.DEVNULL,
                stderr=sv.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
        except Exception as exc:
            if pending_session_id is not None and pending_delete_on_failure:
                manager._delete_durable_session_record(("pi", pending_session_id))
            elif pending_restore_record is not None:
                manager._persist_durable_session_record(pending_restore_record)
            raise RuntimeError(f"spawn failed: {exc}") from exc
        sv.threading.Thread(target=proc.wait, daemon=True).start()
        if pending_session_id is not None:
            sv.threading.Thread(
                target=manager._finalize_pending_pi_spawn,
                kwargs={
                    "spawn_nonce": spawn_nonce,
                    "durable_session_id": pending_session_id,
                    "cwd": cwd3,
                    "session_path": session_path,
                    "proc": proc,
                    "delete_on_failure": pending_delete_on_failure,
                    "restore_record_on_failure": pending_restore_record,
                },
                daemon=True,
            ).start()
            payload = {
                "session_id": pending_session_id,
                "runtime_id": None,
                "backend": "pi",
                "pending_startup": True,
            }
            sv._publish_sessions_invalidate(reason="session_created")
            return payload
        sv._wait_or_raise(proc, label="pi broker", timeout_s=1.5)
        sv._start_proc_stderr_drain(proc)
        meta = sv._wait_for_spawned_broker_meta(spawn_nonce)
        payload = sv._spawn_result_from_meta(meta)
        sv._publish_sessions_invalidate(reason="session_created")
        return payload

    if resume_session_id is not None and worktree_branch is not None:
        raise ValueError("worktree_branch cannot be used when resuming a session")
    spawn_cwd = cwd_path
    if worktree_branch is not None:
        spawn_cwd = sv._create_git_worktree(cwd_path, worktree_branch)

    argv = [sv.sys.executable, "-m", "codoxear.broker", "--cwd", str(spawn_cwd), "--"]
    codex_args: list[str] = []
    resume_row: dict[str, Any] | None = None
    if backend_name == "codex":
        codex_args = [
            "-c",
            sv._codex_trust_override_for_path(spawn_cwd),
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if model is not None:
            codex_args.extend(["--model", model])
        if reasoning_effort is not None:
            codex_args.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        if model_provider is not None:
            codex_args.extend(["-c", f'model_provider="{model_provider}"'])
        if preferred_auth_method is not None:
            codex_args.extend(["-c", f'preferred_auth_method="{preferred_auth_method}"'])
        if service_tier is not None:
            codex_args.extend(["-c", f'service_tier="{service_tier}"'])
    else:
        if preferred_auth_method is not None:
            raise ValueError("preferred_auth_method is not supported for pi")
        if service_tier is not None:
            raise ValueError("service_tier is not supported for pi")
        if model_provider is not None:
            codex_args.extend(["--provider", model_provider])
        if model is not None:
            codex_args.extend(["--model", model])
        if reasoning_effort is not None:
            codex_args.extend(["--thinking", reasoning_effort])
    if resume_session_id is not None:
        resume_id = sv._clean_optional_resume_session_id(resume_session_id)
        if not resume_id:
            raise ValueError("resume_session_id must be a non-empty string")
        found = False
        for row in sv._list_resume_candidates_for_cwd(cwd3, agent_backend=backend_name, limit=1000):
            if row.get("session_id") == resume_id:
                found = True
                resume_row = row
                break
        if not found:
            raise ValueError(f"resume session not found for cwd: {resume_id}")
        if backend_name == "codex":
            codex_args.extend(["resume", resume_id])
        else:
            resume_target = str(resume_row.get("log_path") or "").strip() if isinstance(resume_row, dict) else ""
            codex_args.extend(["--session", resume_target or resume_id])
    codex_args.extend(args or [])
    argv.extend(codex_args)

    env = dict(sv.os.environ)
    if sv._DOTENV.exists():
        for key, value in sv._load_env_file(sv._DOTENV).items():
            env.setdefault(key, value)
    env["CODEX_WEB_OWNER"] = "web"
    env["CODEX_WEB_AGENT_BACKEND"] = backend_name
    if backend_name == "codex":
        env.setdefault("CODEX_HOME", str(sv.CODEX_HOME))
        env.pop("PI_HOME", None)
    else:
        env.setdefault("PI_HOME", str(sv.PI_HOME))
        env.pop("CODEX_HOME", None)
    env.pop("CODEX_WEB_MODEL_PROVIDER", None)
    env.pop("CODEX_WEB_PREFERRED_AUTH_METHOD", None)
    env.pop("CODEX_WEB_MODEL", None)
    env.pop("CODEX_WEB_REASONING_EFFORT", None)
    env.pop("CODEX_WEB_SERVICE_TIER", None)
    env.pop("CODEX_WEB_TRANSPORT", None)
    env.pop("CODEX_WEB_TMUX_SESSION", None)
    env.pop("CODEX_WEB_TMUX_WINDOW", None)
    env.pop("CODEX_WEB_SPAWN_NONCE", None)
    env.pop("CODEX_WEB_RESUME_SESSION_ID", None)
    env.pop("CODEX_WEB_RESUME_LOG_PATH", None)
    spawn_nonce = sv.secrets.token_hex(8)
    env["CODEX_WEB_SPAWN_NONCE"] = spawn_nonce
    if model_provider is not None:
        env["CODEX_WEB_MODEL_PROVIDER"] = model_provider
    if preferred_auth_method is not None:
        env["CODEX_WEB_PREFERRED_AUTH_METHOD"] = preferred_auth_method
    if model is not None:
        env["CODEX_WEB_MODEL"] = model
    if reasoning_effort is not None:
        env["CODEX_WEB_REASONING_EFFORT"] = reasoning_effort
    if service_tier is not None:
        env["CODEX_WEB_SERVICE_TIER"] = service_tier
    if resume_session_id is not None:
        env["CODEX_WEB_RESUME_SESSION_ID"] = resume_session_id
    if create_in_tmux:
        tmux_bin = sv.shutil.which("tmux")
        if tmux_bin is None:
            raise ValueError("tmux is unavailable on this host")
        tmux_window = sv._safe_filename(f"{Path(spawn_cwd).name or 'session'}-{spawn_nonce[:6]}", default="session")
        env["CODEX_WEB_TRANSPORT"] = "tmux"
        env["CODEX_WEB_TMUX_SESSION"] = sv.TMUX_SESSION_NAME
        env["CODEX_WEB_TMUX_WINDOW"] = tmux_window
        env["CODEX_WEB_SPAWN_NONCE"] = spawn_nonce
        short_app_dir = sv._ensure_tmux_short_app_dir()
        inline_env = {
            "CODEX_WEB_OWNER": "web",
            "CODEX_WEB_AGENT_BACKEND": backend_name,
            "CODEX_WEB_TRANSPORT": "tmux",
            "CODEX_WEB_TMUX_SESSION": sv.TMUX_SESSION_NAME,
            "CODEX_WEB_TMUX_WINDOW": tmux_window,
            "CODEX_WEB_SPAWN_NONCE": spawn_nonce,
            "CODOXEAR_APP_DIR": short_app_dir,
        }
        if backend_name == "codex":
            inline_env["CODEX_HOME"] = str(env["CODEX_HOME"])
        else:
            inline_env["PI_HOME"] = str(env["PI_HOME"])
        if resume_session_id is not None:
            inline_env["CODEX_WEB_RESUME_SESSION_ID"] = resume_session_id
        if model_provider is not None:
            inline_env["CODEX_WEB_MODEL_PROVIDER"] = model_provider
        if preferred_auth_method is not None:
            inline_env["CODEX_WEB_PREFERRED_AUTH_METHOD"] = preferred_auth_method
        if model is not None:
            inline_env["CODEX_WEB_MODEL"] = model
        if reasoning_effort is not None:
            inline_env["CODEX_WEB_REASONING_EFFORT"] = reasoning_effort
        if service_tier is not None:
            inline_env["CODEX_WEB_SERVICE_TIER"] = service_tier
        codex_bin = sv._clean_optional_text(sv.os.environ.get("CODEX_BIN"))
        if codex_bin is not None:
            inline_env["CODEX_BIN"] = codex_bin
        repo_root = Path(sv.__file__).resolve().parent.parent
        inline_argv = ["env", *[f"{key}={value}" for key, value in inline_env.items()], *argv]
        shell_cmd = f"cd {sv.shlex.quote(str(repo_root))} && exec {sv.shlex.join(inline_argv)}"
        has_session = sv.subprocess.run(
            [tmux_bin, "has-session", "-t", sv.TMUX_SESSION_NAME],
            stdout=sv.subprocess.DEVNULL,
            stderr=sv.subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if has_session.returncode == 0:
            tmux_argv = [
                tmux_bin,
                "new-window",
                "-d",
                "-P",
                "-F",
                "#{pane_id}",
                "-t",
                f"{sv.TMUX_SESSION_NAME}:",
                "-n",
                tmux_window,
                shell_cmd,
            ]
        else:
            tmux_argv = [
                tmux_bin,
                "new-session",
                "-d",
                "-P",
                "-F",
                "#{pane_id}",
                "-s",
                sv.TMUX_SESSION_NAME,
                "-n",
                tmux_window,
                shell_cmd,
            ]
        tmux_proc = sv.subprocess.run(tmux_argv, capture_output=True, text=True, env=env, check=False)
        if tmux_proc.returncode != 0:
            detail = (tmux_proc.stderr or tmux_proc.stdout or f"exit status {tmux_proc.returncode}").strip()
            raise RuntimeError(f"tmux launch failed: {detail}")
        meta = sv._wait_for_spawned_broker_meta(spawn_nonce)
        payload = sv._spawn_result_from_meta(meta)
        return {**payload, "tmux_session": sv.TMUX_SESSION_NAME, "tmux_window": tmux_window}

    try:
        proc = sv.subprocess.Popen(
            argv,
            stdin=sv.subprocess.DEVNULL,
            stdout=sv.subprocess.DEVNULL,
            stderr=sv.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
    except Exception as exc:
        raise RuntimeError(f"spawn failed: {exc}") from exc

    sv._wait_or_raise(proc, label="broker", timeout_s=1.5)
    if proc.stderr is not None:
        sv.threading.Thread(target=sv._drain_stream, args=(proc.stderr,), daemon=True).start()

    sv.threading.Thread(target=proc.wait, daemon=True).start()
    meta = sv._wait_for_spawned_broker_meta(spawn_nonce)
    payload = sv._spawn_result_from_meta(meta)
    sv._publish_sessions_invalidate(reason="session_created")
    return payload


def restart_session(manager: Any, session_id: str) -> dict[str, Any]:
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        listed_row = _listed_session_row(manager, session_id)
        if isinstance(listed_row, dict) and listed_row.get("pending_startup"):
            raise ValueError("session is still starting")
        raise KeyError("unknown session")
    with manager._lock:
        source = manager._sessions.get(runtime_id)
    if source is None:
        raise KeyError("unknown session")
    if normalize_agent_backend(source.backend, default=source.agent_backend) != "pi":
        raise ValueError("restart is only supported for pi sessions")
    source_path = source.session_path
    if source_path is None or (not source_path.exists()):
        raise ValueError("pi session file not found")
    cwd = _clean_optional_text(source.cwd)
    if cwd is None:
        raise ValueError("session is missing cwd")

    sv = manager._runtime
    durable_session_id = manager._durable_session_id_for_session(source)
    ref = ("pi", durable_session_id)
    create_in_tmux = (source.transport or "").strip().lower() == "tmux"
    preserved_state = manager._capture_runtime_bound_restart_state(runtime_id, ref)
    db = getattr(manager, "_page_state_db", None)
    restore_record = db.load_sessions().get(ref) if isinstance(db, sv.PageStateDB) else None
    if restore_record is None:
        restore_record = sv.DurableSessionRecord(
            backend="pi",
            session_id=durable_session_id,
            cwd=cwd,
            source_path=str(source_path),
            title=source.title,
            first_user_message=source.first_user_message,
            created_at=sv._safe_path_mtime(source_path),
            updated_at=sv._safe_path_mtime(source_path),
            pending_startup=False,
        )
    manager._persist_durable_session_record(
        sv.DurableSessionRecord(
            backend="pi",
            session_id=durable_session_id,
            cwd=restore_record.cwd or cwd,
            source_path=restore_record.source_path or str(source_path),
            title=restore_record.title,
            first_user_message=restore_record.first_user_message,
            created_at=restore_record.created_at,
            updated_at=max(restore_record.updated_at, sv._safe_path_mtime(source_path)),
            pending_startup=True,
        )
    )
    manager._stage_runtime_bound_restart_state(runtime_id, ref, preserved_state)
    if not manager.kill_session(runtime_id):
        manager._restore_runtime_bound_restart_state(runtime_id, ref, preserved_state)
        manager._persist_durable_session_record(restore_record)
        raise RuntimeError("failed to stop source session for restart")
    sv._unlink_quiet(source.sock_path)
    sv._unlink_quiet(source.sock_path.with_suffix(".json"))
    with manager._lock:
        manager._sessions.pop(runtime_id, None)
    sv._publish_sessions_invalidate(reason="session_created")

    provider = _clean_optional_text(source.model_provider)
    model_id = _clean_optional_text(source.model)
    thinking_level = _clean_optional_text(source.reasoning_effort)
    try:
        spawn_res = manager.spawn_web_session(
            cwd=cwd,
            backend="pi",
            resume_session_id=durable_session_id,
            model_provider=provider,
            model=model_id,
            reasoning_effort=thinking_level,
            create_in_tmux=create_in_tmux,
        )
    except Exception:
        manager._persist_durable_session_record(restore_record)
        sv._publish_sessions_invalidate(reason="session_created")
        raise

    payload = dict(spawn_res)
    payload["session_id"] = durable_session_id
    payload["backend"] = "pi"
    payload["previous_runtime_id"] = runtime_id
    launched_runtime_id = _clean_optional_text(payload.get("runtime_id"))
    if launched_runtime_id is not None:
        manager._restore_runtime_bound_restart_state(launched_runtime_id, ref, preserved_state)
        manager._persist_durable_session_record(
            sv.DurableSessionRecord(
                backend="pi",
                session_id=durable_session_id,
                cwd=restore_record.cwd or cwd,
                source_path=restore_record.source_path or str(source_path),
                title=restore_record.title,
                first_user_message=restore_record.first_user_message,
                created_at=restore_record.created_at,
                updated_at=max(restore_record.updated_at, sv._safe_path_mtime(source_path)),
                pending_startup=False,
            )
        )
    else:
        sv.threading.Thread(
            target=manager._finalize_pending_pi_restart_state,
            kwargs={
                "durable_session_id": durable_session_id,
                "ref": ref,
                "state": preserved_state,
            },
            daemon=True,
        ).start()
    return payload


def handoff_session(manager: Any, session_id: str) -> dict[str, Any]:
    runtime_id = manager._runtime_session_id_for_identifier(session_id)
    if runtime_id is None:
        listed_row = _listed_session_row(manager, session_id)
        if isinstance(listed_row, dict) and listed_row.get("pending_startup"):
            raise ValueError("session is still starting")
        raise KeyError("unknown session")
    with manager._lock:
        source = manager._sessions.get(runtime_id)
    if source is None:
        raise KeyError("unknown session")
    if normalize_agent_backend(source.backend, default=source.agent_backend) != "pi":
        raise ValueError("handoff is only supported for pi sessions")
    source_path = source.session_path
    if source_path is None or (not source_path.exists()):
        raise ValueError("pi session file not found")
    cwd = _clean_optional_text(source.cwd)
    if cwd is None:
        raise ValueError("session is missing cwd")

    sv = manager._runtime
    source_session_id = manager._durable_session_id_for_session(source)
    history_path = sv._next_pi_handoff_history_path(source_path)
    new_session_id = str(sv.uuid.uuid4())
    new_session_path = sv._pi_new_session_file_for_cwd(cwd)
    provider, model_id, thinking_level = sv._read_pi_run_settings(source_path)
    provider = _clean_optional_text(source.model_provider) or provider
    model_id = _clean_optional_text(source.model) or model_id
    thinking_level = _clean_optional_text(source.reasoning_effort) or thinking_level
    create_in_tmux = (source.transport or "").strip().lower() == "tmux"
    copied_history = False
    launched_session_id = new_session_id
    launched_runtime_id: str | None = None
    try:
        sv._copy_file_atomic(source_path, history_path)
        copied_history = True
        sv._write_pi_handoff_session(
            new_session_path,
            session_id=new_session_id,
            cwd=cwd,
            source_session_id=source_session_id,
            history_path=history_path,
            provider=provider,
            model_id=model_id,
            thinking_level=thinking_level,
        )
        spawn_res = manager.spawn_web_session(
            cwd=cwd,
            backend="pi",
            resume_session_id=new_session_id,
            model_provider=provider,
            model=model_id,
            reasoning_effort=thinking_level,
            create_in_tmux=create_in_tmux,
        )
        launched_session_id = _clean_optional_text(spawn_res.get("session_id")) or new_session_id
        launched = manager._wait_for_live_session(launched_session_id)
        launched_runtime_id = launched.session_id
        alias = manager._copy_session_ui_identity(
            source_session_id=session_id,
            target_session_id=launched_session_id,
        )
        if not manager.delete_session(runtime_id):
            raise RuntimeError("failed to stop source session after handoff launch")
        payload = dict(spawn_res)
        payload["session_id"] = launched_session_id
        payload["runtime_id"] = launched_runtime_id
        payload["backend"] = "pi"
        payload["history_path"] = str(history_path)
        payload["previous_session_id"] = source_session_id
        if alias:
            payload["alias"] = alias
        return payload
    except Exception:
        if launched_runtime_id is not None:
            try:
                manager.delete_session(launched_runtime_id)
            except Exception:
                pass
        else:
            try:
                manager.delete_session(launched_session_id)
            except Exception:
                pass
        sv._unlink_quiet(new_session_path)
        if copied_history:
            sv._unlink_quiet(history_path)
        raise
