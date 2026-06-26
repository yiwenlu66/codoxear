from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from codoxear.session_discovery import DiscoveryDeps
from codoxear.session_discovery import discover_sessions


def _write_sidecar(sock: Path, *, root: Path, log_path: Path | None, **overrides) -> dict:
    meta = {
        "session_id": sock.stem,
        "agent_backend": "codex",
        "owner": "web",
        "broker_pid": 11,
        "codex_pid": 12,
        "cwd": str(root),
        "start_ts": 123.0,
        "updated_ts": 456.0,
        "log_path": str(log_path) if log_path is not None else None,
        "model_provider": "openai",
        "preferred_auth_method": "apikey",
        "model": "gpt-test",
        "reasoning_effort": "low",
        "service_tier": "flex",
        "control_protocol_version": 2,
        "control_capabilities": {"sync_send": True, "key_write_errors": True},
    }
    meta.update(overrides)
    sock.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _deps(
    *,
    live_pids: set[int] | None = None,
    sock_response: dict | None = None,
    sock_error: BaseException | None = None,
    proc_log_path: Path | None = None,
    meta_payload: dict | None = None,
    coerced: tuple[str, Path] | None = None,
):
    live = set(live_pids or set())

    def sock_call(_sock: Path, _req: dict, _timeout_s: float) -> dict:
        if sock_error is not None:
            raise sock_error
        return dict(sock_response or {"busy": False, "queue_len": 0, "interrupted_idle": False, "token": None})

    return DiscoveryDeps(
        pid_alive=lambda pid: int(pid) in live,
        proc_find_open_rollout_log=lambda _proc_root, _root_pid, _agent_backend, _cwd, _ignored_paths: proc_log_path,
        read_session_meta_or_none=lambda _log_path, _agent_backend, _context: meta_payload,
        coerce_main_thread_log=lambda thread_id, log_path: coerced if coerced is not None else (thread_id, log_path),
        session_transport=lambda meta: (meta.get("transport"), meta.get("tmux_session"), meta.get("tmux_window")),
        session_run_settings=lambda meta, _log_path, _agent_backend: (
            meta.get("model_provider"),
            meta.get("preferred_auth_method"),
            meta.get("model"),
            meta.get("reasoning_effort"),
        ),
        sock_call=sock_call,
        broker_busy_queue_from_state=lambda state: (bool(state.get("busy")), int(state.get("queue_len", 0))),
        broker_interrupted_idle_from_state=lambda state: bool(state.get("interrupted_idle")),
        sock_error_definitely_stale=lambda exc: isinstance(exc, FileNotFoundError),
        token_update_finder=lambda _log_path: {"total": 7},
    )


def test_session_discovery_import_does_not_import_server() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import codoxear.session_discovery; raise SystemExit(1 if 'codoxear.server' in sys.modules else 0)",
        ],
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0


def test_discover_valid_registration_from_sidecar(tmp_path: Path) -> None:
    sock_dir = tmp_path / "socks"
    sock_dir.mkdir()
    sock = sock_dir / "session-a.sock"
    sock.touch()
    log_path = tmp_path / "rollout.jsonl"
    log_path.write_text('{"type":"session_meta","payload":{"id":"thread-a"}}\n', encoding="utf-8")
    _write_sidecar(sock, root=tmp_path, log_path=log_path, session_id="sidecar-thread")

    result = discover_sessions(
        sock_dir,
        proc_root=tmp_path / "proc",
        hidden_sessions=set(),
        deps=_deps(meta_payload={"id": "log-thread"}),
    )

    assert result.stale_actions == []
    assert [(recent.cwd, recent.ts) for recent in result.recent_cwds] == [(str(tmp_path), 456.0)]
    assert len(result.registrations) == 1
    reg = result.registrations[0]
    assert reg.session_id == "session-a"
    assert reg.thread_id == "log-thread"
    assert reg.broker_pid == 11
    assert reg.codex_pid == 12
    assert reg.agent_backend == "codex"
    assert reg.owned is True
    assert reg.cwd == str(tmp_path)
    assert reg.log_path == log_path
    assert reg.meta_log_off == log_path.stat().st_size
    assert reg.token == {"total": 7}
    assert reg.model_provider == "openai"
    assert reg.preferred_auth_method == "apikey"
    assert reg.model == "gpt-test"
    assert reg.reasoning_effort == "low"
    assert reg.service_tier == "flex"
    assert reg.sync_send_supported is True
    assert reg.key_write_errors_supported is True


def test_discover_missing_metadata_emits_clear_state_action(tmp_path: Path) -> None:
    sock_dir = tmp_path / "socks"
    sock_dir.mkdir()
    sock = sock_dir / "orphan.sock"
    sock.touch()

    result = discover_sessions(sock_dir, proc_root=tmp_path / "proc", hidden_sessions=set(), deps=_deps())

    assert result.registrations == []
    assert len(result.stale_actions) == 1
    action = result.stale_actions[0]
    assert action.session_id == "orphan"
    assert action.sock_path == sock
    assert action.clear_session_state is True
    assert action.unhide_session is True
    assert action.failure_record is None


def test_discover_malformed_sidecar_is_not_pruned(tmp_path: Path) -> None:
    sock_dir = tmp_path / "socks"
    sock_dir.mkdir()
    sock = sock_dir / "bad.sock"
    sock.touch()
    sock.with_suffix(".json").write_text("{not-json}\n", encoding="utf-8")

    result = discover_sessions(sock_dir, proc_root=tmp_path / "proc", hidden_sessions=set(), deps=_deps())

    assert result.registrations == []
    assert result.stale_actions == []


def test_discover_dead_owned_without_log_records_failure_without_clearing_state(tmp_path: Path) -> None:
    sock_dir = tmp_path / "socks"
    sock_dir.mkdir()
    sock = sock_dir / "dead.sock"
    sock.touch()
    _write_sidecar(sock, root=tmp_path, log_path=None, launch_id="launch-1", spawn_nonce="nonce-1")

    result = discover_sessions(sock_dir, proc_root=tmp_path / "proc", hidden_sessions=set(), deps=_deps(live_pids=set()))

    assert result.registrations == []
    assert len(result.stale_actions) == 1
    action = result.stale_actions[0]
    assert action.clear_session_state is False
    assert action.unhide_session is True
    assert action.failure_record is not None
    assert action.failure_record["launch_id"] == "launch-1"
    assert action.failure_record["spawn_nonce"] == "nonce-1"
    assert action.failure_record["stage"] == "broker_exit_before_log_bind"


def test_discover_hidden_live_session_is_excluded_without_cleanup(tmp_path: Path) -> None:
    sock_dir = tmp_path / "socks"
    sock_dir.mkdir()
    sock = sock_dir / "hidden.sock"
    sock.touch()
    log_path = tmp_path / "hidden.jsonl"
    log_path.write_text("{}\n", encoding="utf-8")
    _write_sidecar(sock, root=tmp_path, log_path=log_path)

    result = discover_sessions(
        sock_dir,
        proc_root=tmp_path / "proc",
        hidden_sessions={"hidden"},
        deps=_deps(live_pids={11, 12}, meta_payload={"id": "hidden"}),
    )

    assert result.registrations == []
    assert result.recent_cwds == []
    assert result.stale_actions == []


def test_discover_live_unresponsive_socket_is_excluded_without_cleanup(tmp_path: Path) -> None:
    sock_dir = tmp_path / "socks"
    sock_dir.mkdir()
    sock = sock_dir / "starting.sock"
    sock.touch()
    _write_sidecar(sock, root=tmp_path, log_path=None, owner="terminal")

    result = discover_sessions(
        sock_dir,
        proc_root=tmp_path / "proc",
        hidden_sessions=set(),
        deps=_deps(live_pids={11, 12}, sock_error=FileNotFoundError("startup race")),
    )

    assert result.registrations == []
    assert [(recent.cwd, recent.ts) for recent in result.recent_cwds] == [(str(tmp_path), 456.0)]
    assert result.stale_actions == []
