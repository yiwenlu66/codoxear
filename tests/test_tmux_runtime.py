from __future__ import annotations

import subprocess
import sys

from codoxear.tmux_runtime import tmux_pane_snapshot


def test_tmux_runtime_import_does_not_load_server() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; import codoxear.tmux_runtime; raise SystemExit('codoxear.server' in sys.modules)"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_tmux_pane_snapshot_returns_empty_without_target() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    assert tmux_pane_snapshot("/usr/bin/tmux", tmux_session_name="codoxear", run=run) == {}
    assert calls == []


def test_tmux_pane_snapshot_uses_pane_id_and_captures_tail() -> None:
    calls: list[list[str]] = []
    tail = "a" * 4100

    def run(argv: list[str], **kwargs):
        calls.append(argv)
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        if argv[1] == "display-message":
            return subprocess.CompletedProcess(argv, 0, stdout="%8\t123\t0\t\tpython\twork\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=tail, stderr="")

    snapshot = tmux_pane_snapshot("/usr/bin/tmux", tmux_session_name="codoxear", pane_id=" %8 ", window="ignored", run=run)

    assert calls == [
        [
            "/usr/bin/tmux",
            "display-message",
            "-p",
            "-t",
            "%8",
            "#{pane_id}\t#{pane_pid}\t#{pane_dead}\t#{pane_dead_status}\t#{pane_current_command}\t#{window_name}",
        ],
        ["/usr/bin/tmux", "capture-pane", "-p", "-t", "%8", "-S", "-80"],
    ]
    assert snapshot == {
        "tmux_target": "%8",
        "tmux_pane_id": "%8",
        "tmux_pane_pid": "123",
        "tmux_pane_dead": "0",
        "tmux_pane_dead_status": "",
        "tmux_pane_command": "python",
        "tmux_window": "work",
        "tmux_pane_tail": "a" * 4000,
    }


def test_tmux_pane_snapshot_uses_window_fallback_with_session_name() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs):
        calls.append(argv)
        if argv[1] == "display-message":
            return subprocess.CompletedProcess(argv, 0, stdout="%9\t456\t1\t42\tbash\twin\n", stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="capture failed")

    snapshot = tmux_pane_snapshot("tmux", tmux_session_name="session-a", pane_id="  ", window=" win ", run=run)

    assert calls[0][4] == "session-a: win "
    assert calls[1][4] == "session-a: win "
    assert snapshot["tmux_target"] == "session-a: win "
    assert snapshot["tmux_pane_dead_status"] == "42"
    assert snapshot["tmux_capture_error"] == "capture failed"


def test_tmux_pane_snapshot_returns_inspect_error_without_capture() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr="display failed\n")

    snapshot = tmux_pane_snapshot("tmux", tmux_session_name="codoxear", window="work", run=run)

    assert len(calls) == 1
    assert snapshot == {"tmux_target": "codoxear:work", "tmux_inspect_error": "display failed"}


def test_tmux_pane_snapshot_uses_exit_status_when_error_streams_empty() -> None:
    def run(argv: list[str], **kwargs):
        if argv[1] == "display-message":
            return subprocess.CompletedProcess(argv, 0, stdout="%9\t456\t0\t\tbash\twin\n", stderr="")
        return subprocess.CompletedProcess(argv, 3, stdout="", stderr="")

    snapshot = tmux_pane_snapshot("tmux", tmux_session_name="codoxear", pane_id="%9", run=run)

    assert snapshot["tmux_capture_error"] == "exit status 3"
