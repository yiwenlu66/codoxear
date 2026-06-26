from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from codoxear.launch_ledger import LaunchAttemptRecorder
from codoxear.session_launcher import LaunchProcessDeps
from codoxear.session_launcher import LaunchProcessFailure
from codoxear.session_launcher import LaunchProcessRequest
from codoxear.session_launcher import launch_broker_process


def _request(*, create_in_tmux: bool = False) -> LaunchProcessRequest:
    return LaunchProcessRequest(
        argv=[sys.executable, "-m", "codoxear.broker", "--cwd", "/tmp/work", "--", "--flag"],
        env={"CODEX_WEB_AGENT_BACKEND": "codex", "CODEX_HOME": "/tmp/codex-home"},
        agent_backend="codex",
        spawn_cwd=Path("/tmp/work"),
        launch_id="launch-test",
        spawn_nonce="nonce1234567890",
        create_in_tmux=create_in_tmux,
        tmux_session_name="codoxear-test",
        repo_root=Path("/repo"),
        model_provider="provider-a",
        preferred_auth_method="apikey",
        model="gpt-test",
        reasoning_effort="low",
        service_tier="fast",
    )


def _recorder(records: list[dict]) -> LaunchAttemptRecorder:
    counter = {"value": 10.0}

    def now() -> float:
        counter["value"] += 1.0
        return counter["value"]

    def persist(record: dict) -> dict:
        rec = dict(record)
        records.append(rec)
        return rec

    return LaunchAttemptRecorder({"launch_id": "launch-test", "created_ts": 1.0}, record_launch_attempt=persist, now=now)


def _deps(**overrides) -> LaunchProcessDeps:
    values = {
        "which_tmux": lambda _name: "/usr/bin/tmux",
        "run": lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="%8\n", stderr=""),
        "popen": lambda *args, **kwargs: SimpleNamespace(pid=1234, stderr=None, wait=lambda: 0),
        "wait_or_raise": lambda *args, **kwargs: None,
        "wait_for_spawned_broker_meta": lambda _nonce: {"broker_pid": 5678},
        "tmux_pane_snapshot": lambda *args, **kwargs: {"tmux_pane_id": "%8", "tmux_pane_dead": "0", "tmux_window": "work-nonce"},
        "drain_stream": lambda _stream: None,
    }
    values.update(overrides)
    return LaunchProcessDeps(**values)


def test_session_launcher_import_does_not_load_server() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; import codoxear.session_launcher; raise SystemExit('codoxear.server' in sys.modules)"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_launch_broker_process_direct_records_spawn_and_starts_wait_thread() -> None:
    records: list[dict] = []
    popen_calls: list[tuple[tuple, dict]] = []
    thread_targets: list[object] = []

    class Proc:
        pid = 2468
        stderr = None

        def wait(self) -> int:
            return 0

    def popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return Proc()

    with patch.object(threading.Thread, "start", lambda self: thread_targets.append(getattr(self, "_target", None))):
        result = launch_broker_process(_request(), recorder=_recorder(records), deps=_deps(popen=popen))

    assert result == {"broker_pid": 2468}
    assert [record["state"] for record in records] == ["starting", "broker_spawned"]
    assert records[0]["transport"] == "direct"
    assert records[1]["transport"] == "direct"
    assert records[1]["broker_pid"] == 2468
    assert popen_calls[0][0][0] == _request().argv
    assert popen_calls[0][1]["env"]["CODEX_WEB_AGENT_BACKEND"] == "codex"
    assert popen_calls[0][1]["start_new_session"] is True
    assert len(thread_targets) == 1


def test_launch_broker_process_direct_failure_raises_record() -> None:
    records: list[dict] = []

    class Proc:
        pid = 2469
        stderr = None

        def wait(self) -> int:
            return 0

    def wait_or_raise(*args, **kwargs) -> None:
        raise RuntimeError("broker exited early (rc=1): boom")

    with pytest.raises(LaunchProcessFailure) as err:
        launch_broker_process(
            _request(),
            recorder=_recorder(records),
            deps=_deps(popen=lambda *args, **kwargs: Proc(), wait_or_raise=wait_or_raise),
        )

    assert [record["state"] for record in records] == ["starting", "failed"]
    assert err.value.record["state"] == "failed"
    assert err.value.record["stage"] == "broker_early_exit"
    assert "boom" in err.value.record["error"]
    assert err.value.record["broker_pid"] == 2469
    assert err.value.record["transport"] == "direct"


def test_launch_broker_process_tmux_metadata_delay_returns_pending_and_records_context() -> None:
    records: list[dict] = []
    run_calls: list[tuple[tuple, dict]] = []
    request = _request(create_in_tmux=True)

    def run(*args, **kwargs):
        run_calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, stdout="%8\n", stderr="")

    def wait_meta(_nonce: str) -> dict:
        raise TimeoutError("metadata not ready")

    result = launch_broker_process(
        request,
        recorder=_recorder(records),
        deps=_deps(run=run, wait_for_spawned_broker_meta=wait_meta),
    )

    assert result["pending"] is True
    assert result["launch_id"] == "launch-test"
    assert result["tmux_session"] == "codoxear-test"
    assert result["tmux_window"].startswith("work-")
    assert request.env["CODEX_WEB_TRANSPORT"] == "tmux"
    assert request.env["CODEX_WEB_TMUX_SESSION"] == "codoxear-test"
    assert request.env["CODEX_WEB_TMUX_WINDOW"] == result["tmux_window"]
    assert [record["state"] for record in records] == ["starting", "tmux_pane_created", "tmux_pane_created"]
    assert records[-1]["stage"] == "broker_metadata_pending"
    assert records[-1]["error"] == "metadata not ready"
    tmux_argv = run_calls[0][0][0]
    assert tmux_argv[:8] == ["/usr/bin/tmux", "new-window", "-d", "-P", "-F", "#{pane_id}", "-t", "codoxear-test:"]
    shell_cmd = tmux_argv[-1]
    assert "CODEX_WEB_TRANSPORT=tmux" in shell_cmd
    assert "CODEX_WEB_LAUNCH_ID=launch-test" in shell_cmd
    assert "unset CODEX_HOME PI_HOME CLAUDE_CONFIG_DIR" in shell_cmd
