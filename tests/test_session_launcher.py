from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from codoxear.launch_ledger import LaunchAttemptRecorder
from codoxear.session_launcher import LaunchProcessDeps
from codoxear.session_launcher import LaunchProcessFailure
from codoxear.session_launcher import LaunchProcessRequest
from codoxear.session_launcher import launch_broker_process
from codoxear.session_launcher import wait_for_spawned_broker_meta


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


def test_launch_broker_process_direct_spawn_failure_raises_record() -> None:
    records: list[dict] = []

    def popen(*args, **kwargs):
        raise OSError("no broker")

    with pytest.raises(LaunchProcessFailure) as err:
        launch_broker_process(_request(), recorder=_recorder(records), deps=_deps(popen=popen))

    assert [record["state"] for record in records] == ["starting", "failed"]
    assert err.value.record["stage"] == "broker_spawn"
    assert err.value.record["transport"] == "direct"
    assert err.value.record["error"] == "spawn failed: no broker"


def test_launch_broker_process_direct_stderr_starts_drain_before_wait_thread() -> None:
    records: list[dict] = []
    stderr_stream = object()
    thread_targets: list[object] = []
    thread_args: list[tuple] = []

    class Proc:
        pid = 2470

        def __init__(self) -> None:
            self.stderr = stderr_stream

        def wait(self) -> int:
            return 0

    def drain_stream(stream: object) -> None:
        assert stream is stderr_stream

    def start_thread(self) -> None:
        thread_targets.append(getattr(self, "_target", None))
        thread_args.append(getattr(self, "_args", ()))

    with patch.object(threading.Thread, "start", start_thread):
        result = launch_broker_process(
            _request(),
            recorder=_recorder(records),
            deps=_deps(popen=lambda *args, **kwargs: Proc(), drain_stream=drain_stream),
        )

    assert result == {"broker_pid": 2470}
    assert [record["state"] for record in records] == ["starting", "broker_spawned"]
    assert thread_targets[0] is drain_stream
    assert getattr(thread_targets[1], "__name__", "") == "wait"
    assert thread_args[0] == (stderr_stream,)
    assert thread_args[1] == ()


def test_launch_broker_process_tmux_success_records_broker_meta_bound() -> None:
    records: list[dict] = []
    result = launch_broker_process(_request(create_in_tmux=True), recorder=_recorder(records), deps=_deps())

    assert result == {"broker_pid": 5678, "tmux_session": "codoxear-test", "tmux_window": "work-nonce1"}
    assert [record["state"] for record in records] == ["starting", "tmux_pane_created", "broker_meta_bound"]
    assert records[1]["tmux_attempts"] == [{"cmd": "new-window", "returncode": 0, "stderr": "", "stdout": "%8"}]
    assert records[2]["transport"] == "tmux"
    assert records[2]["tmux_pane_id"] == "%8"
    assert records[2]["broker_pid"] == 5678


def test_launch_broker_process_tmux_missing_session_duplicate_retry_then_success() -> None:
    records: list[dict] = []
    run_results = iter(
        [
            subprocess.CompletedProcess(["tmux", "new-window"], 1, stdout="", stderr="can't find session: codoxear-test"),
            subprocess.CompletedProcess(["tmux", "new-session"], 1, stdout="", stderr="duplicate session: codoxear-test"),
            subprocess.CompletedProcess(["tmux", "new-window"], 0, stdout="%9\n", stderr=""),
        ]
    )
    run_cmds: list[str] = []

    def run(argv: list[str], **kwargs):
        run_cmds.append(argv[1])
        return next(run_results)

    result = launch_broker_process(_request(create_in_tmux=True), recorder=_recorder(records), deps=_deps(run=run))

    assert result["broker_pid"] == 5678
    assert run_cmds == ["new-window", "new-session", "new-window"]
    assert records[1]["tmux_attempts"] == [
        {"cmd": "new-window", "returncode": 1, "stderr": "can't find session: codoxear-test", "stdout": ""},
        {"cmd": "new-session", "returncode": 1, "stderr": "duplicate session: codoxear-test", "stdout": ""},
        {"cmd": "new-window-after-duplicate", "returncode": 0, "stderr": "", "stdout": "%9"},
    ]


def test_launch_broker_process_tmux_launch_failure_raises_record() -> None:
    records: list[dict] = []

    def run(argv: list[str], **kwargs):
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr="tmux refused")

    with pytest.raises(LaunchProcessFailure) as err:
        launch_broker_process(_request(create_in_tmux=True), recorder=_recorder(records), deps=_deps(run=run))

    assert [record["state"] for record in records] == ["starting", "failed"]
    assert err.value.record["stage"] == "tmux_launch"
    assert err.value.record["transport"] == "tmux"
    assert err.value.record["tmux_exit_status"] == 2
    assert err.value.record["tmux_stderr"] == "tmux refused"
    assert err.value.record["tmux_attempts"] == [{"cmd": "new-window", "returncode": 2, "stderr": "tmux refused", "stdout": ""}]


def test_launch_broker_process_tmux_missing_broker_pid_raises_record() -> None:
    records: list[dict] = []

    with pytest.raises(LaunchProcessFailure) as err:
        launch_broker_process(
            _request(create_in_tmux=True),
            recorder=_recorder(records),
            deps=_deps(wait_for_spawned_broker_meta=lambda _nonce: {"broker_pid": "5678"}),
        )

    assert [record["state"] for record in records] == ["starting", "tmux_pane_created", "failed"]
    assert err.value.record["stage"] == "broker_metadata"
    assert err.value.record["transport"] == "tmux"
    assert err.value.record["metadata"] == {"broker_pid": "5678"}


def test_launch_broker_process_tmux_dead_pane_metadata_failure_raises_with_fresh_snapshot() -> None:
    records: list[dict] = []
    snapshots = iter(
        [
            {"tmux_pane_id": "%8", "tmux_pane_dead": "1", "tmux_window": "work-nonce"},
            {"tmux_pane_id": "%8", "tmux_pane_dead": "1", "tmux_pane_dead_status": "42", "tmux_window": "work-nonce"},
        ]
    )
    snapshot_calls: list[dict] = []

    def snapshot(*args, **kwargs):
        snapshot_calls.append(kwargs)
        return next(snapshots)

    with pytest.raises(LaunchProcessFailure) as err:
        launch_broker_process(
            _request(create_in_tmux=True),
            recorder=_recorder(records),
            deps=_deps(tmux_pane_snapshot=snapshot, wait_for_spawned_broker_meta=lambda _nonce: (_ for _ in ()).throw(TimeoutError("metadata not ready"))),
        )

    assert [record["state"] for record in records] == ["starting", "tmux_pane_created", "failed"]
    assert len(snapshot_calls) == 2
    assert err.value.record["stage"] == "broker_metadata"
    assert err.value.record["tmux_pane_dead_status"] == "42"
    assert err.value.record["error"] == "metadata not ready"


def test_wait_for_spawned_broker_meta_skips_invalid_metadata_until_live_match() -> None:
    with TemporaryDirectory() as td:
        sock_dir = Path(td)
        for name in ["a-bad-json", "b-other-nonce", "c-dead-pid", "d-good"]:
            (sock_dir / f"{name}.json").write_text("{}", encoding="utf-8")
        metas = {
            "b-other-nonce.json": {"spawn_nonce": "other", "broker_pid": 111},
            "c-dead-pid.json": {"spawn_nonce": "target", "broker_pid": 0},
            "d-good.json": {"spawn_nonce": "target", "broker_pid": 222},
        }
        read_paths: list[str] = []
        live_checks: list[tuple[int, str]] = []

        def read_metadata(path: Path, *, sock: Path) -> dict:
            read_paths.append(path.name)
            if path.name == "a-bad-json.json":
                raise ValueError("bad json")
            return dict(metas[path.name])

        def required_live_pid(meta: dict, key: str, *, sock: Path) -> int:
            pid = meta[key]
            live_checks.append((pid, sock.name))
            if pid <= 0:
                raise ValueError("dead pid")
            return int(pid)

        meta = wait_for_spawned_broker_meta(
            "target",
            sock_dir=sock_dir,
            timeout_s=0.0,
            now=lambda: 0.0,
            read_metadata=read_metadata,
            required_live_pid=required_live_pid,
            sleep=lambda _seconds: None,
        )

    assert meta == {"spawn_nonce": "target", "broker_pid": 222}
    assert read_paths == ["a-bad-json.json", "b-other-nonce.json", "c-dead-pid.json", "d-good.json"]
    assert live_checks == [(0, "c-dead-pid.sock"), (222, "d-good.sock")]


def test_wait_for_spawned_broker_meta_timeout_uses_bounded_poll_loop() -> None:
    now_values = iter([0.0, 0.0, 0.1])
    sleeps: list[float] = []
    with TemporaryDirectory() as td:
        with pytest.raises(RuntimeError, match="tmux launch did not publish broker metadata within 0.0s"):
            wait_for_spawned_broker_meta(
                "missing",
                sock_dir=Path(td),
                timeout_s=0.0,
                now=lambda: next(now_values),
                sleep=lambda seconds: sleeps.append(seconds),
            )

    assert sleeps == [0.05]
