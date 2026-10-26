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


# -- window listing / swapping (sidebar-order mirror) -------------------


def test_parse_window_rows_keeps_window_names_containing_tabs() -> None:
    from codoxear.tmux_runtime import parse_window_rows

    rows = parse_window_rows("2\t@5\t1\t\tmy\twindow\n0\t@7\t0\tabc123abc123\tplain\n\nbogus\n")

    assert rows == [
        {"index": 0, "id": "@7", "active": False, "nonce": "abc123abc123", "name": "plain"},
        {"index": 2, "id": "@5", "active": True, "nonce": None, "name": "my\twindow"},
    ]


def test_parse_window_rows_rejects_an_unexpanded_user_option_as_identity() -> None:
    # A tmux that cannot expand user options returns the format text verbatim;
    # treating that as a token would make every window claim the same identity.
    from codoxear.tmux_runtime import parse_window_rows

    rows = parse_window_rows("0\t@1\t0\t#{@codoxear_nonce}\tplain\n")

    assert rows == [{"index": 0, "id": "@1", "active": False, "nonce": None, "name": "plain"}]


def test_tmux_list_windows_raises_with_the_tmux_diagnostic() -> None:
    from codoxear.tmux_runtime import TmuxCommandError, tmux_list_windows

    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs):
        calls.append(argv)
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="can't find session: codoxear\n")

    try:
        tmux_list_windows("/usr/bin/tmux", tmux_session_name="codoxear", run=run)
    except TmuxCommandError as exc:
        assert exc.command == "list-windows"
        assert "can't find session" in exc.detail
    else:  # pragma: no cover - the call must fail loudly
        raise AssertionError("expected TmuxCommandError")

    assert calls == [
        [
            "/usr/bin/tmux",
            "list-windows",
            "-t",
            "codoxear",
            "-F",
            "#{window_index}\t#{window_id}\t#{window_active}\t#{@codoxear_nonce}\t#{window_name}",
        ]
    ]


def test_tmux_swap_windows_never_passes_a_detach_flag() -> None:
    # -d means different things on tmux 3.5a and 3.8, so the mirror restores the
    # viewer's window explicitly instead of relying on it.
    from codoxear.tmux_runtime import tmux_swap_windows

    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    tmux_swap_windows("tmux", src_window_id="@1", dst_window_id="@4", run=run)

    assert calls == [["tmux", "swap-window", "-s", "@1", "-t", "@4"]]


def test_tmux_swap_windows_fails_loudly_when_tmux_rejects_the_swap() -> None:
    from codoxear.tmux_runtime import TmuxCommandError, tmux_swap_windows

    def run(argv: list[str], **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="can't find window: @9\n")

    try:
        tmux_swap_windows("tmux", src_window_id="@1", dst_window_id="@9", run=run)
    except TmuxCommandError as exc:
        assert exc.command == "swap-window"
        assert exc.detail == "can't find window: @9"
    else:  # pragma: no cover - the call must fail loudly
        raise AssertionError("expected TmuxCommandError")


def test_tmux_select_window_targets_one_session_by_window_id() -> None:
    from codoxear.tmux_runtime import tmux_select_window

    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    tmux_select_window("tmux", window_id="@7", tmux_session_name="codoxear", run=run)

    assert calls == [["tmux", "select-window", "-t", "codoxear:@7"]]


def test_tmux_select_window_fails_loudly_when_the_window_is_gone() -> None:
    from codoxear.tmux_runtime import TmuxCommandError, tmux_select_window

    def run(argv: list[str], **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="can't find window: codoxear:@7\n")

    try:
        tmux_select_window("tmux", window_id="@7", tmux_session_name="codoxear", run=run)
    except TmuxCommandError as exc:
        assert exc.command == "select-window"
    else:  # pragma: no cover
        raise AssertionError("expected TmuxCommandError")


def test_tmux_window_naming_and_identity_commands() -> None:
    from codoxear.tmux_runtime import tmux_rename_window, tmux_renumber_windows, tmux_stamp_window_nonce

    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    tmux_stamp_window_nonce("tmux", window_id="@3", nonce="abc123", run=run)
    tmux_rename_window("tmux", window_id="@3", name="my session", run=run)
    tmux_renumber_windows("tmux", tmux_session_name="codoxear", run=run)

    assert calls == [
        ["tmux", "set-option", "-w", "-t", "@3", "@codoxear_nonce", "abc123"],
        ["tmux", "rename-window", "-t", "@3", "my session"],
        ["tmux", "move-window", "-r", "-t", "codoxear"],
    ]


def test_tmux_base_index_prefers_the_session_value_and_tolerates_failure() -> None:
    from codoxear.tmux_runtime import tmux_base_index

    def run_session_override(argv: list[str], **kwargs):
        if argv[1:3] == ["show-options", "-v"]:
            return subprocess.CompletedProcess(argv, 0, stdout="1\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="0\n", stderr="")

    def run_global_only(argv: list[str], **kwargs):
        if argv[1:3] == ["show-options", "-v"]:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="2\n", stderr="")

    def run_broken(argv: list[str], **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unknown option")

    assert tmux_base_index("tmux", tmux_session_name="codoxear", run=run_session_override) == 1
    assert tmux_base_index("tmux", tmux_session_name="codoxear", run=run_global_only) == 2
    assert tmux_base_index("tmux", tmux_session_name="codoxear", run=run_broken) is None
