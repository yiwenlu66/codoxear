import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from codoxear.broker import Broker
from codoxear.broker import State


def _broker_state(*, codex_pid: int, sock_path: Path) -> State:
    return State(
        codex_pid=codex_pid,
        pty_master_fd=1,
        cwd="/tmp",
        start_ts=0.0,
        codex_home=Path("/tmp"),
        sessions_dir=Path("/tmp"),
        sock_path=sock_path,
    )


class _AcceptCrashSocket:
    def bind(self, _addr: str) -> None:
        return

    def listen(self, _backlog: int) -> None:
        return

    def settimeout(self, _timeout: float) -> None:
        return

    def accept(self) -> tuple[object, object]:
        raise RuntimeError("boom")

    def close(self) -> None:
        return


class _FakeThread:
    started_targets: list[str] = []

    def __init__(self, *, target, daemon: bool) -> None:
        self._target = target
        self._daemon = daemon

    def start(self) -> None:
        self.started_targets.append(self._target.__name__)


class TestBrokerFailClosed(unittest.TestCase):
    def test_teardown_managed_process_group_kills_real_process_group(self) -> None:
        proc = subprocess.Popen(["sh", "-c", "sleep 100"], start_new_session=True)
        try:
            broker = Broker(cwd="/tmp", codex_args=[])
            broker.state = _broker_state(codex_pid=proc.pid, sock_path=Path("/tmp/test-broker.sock"))

            broker._teardown_managed_process_group(wait_seconds=0.2)

            proc.wait(timeout=2.0)
            self.assertIsNotNone(proc.returncode)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)

    def test_discover_log_watcher_failure_triggers_teardown(self) -> None:
        broker = Broker(cwd="/tmp", codex_args=[])
        broker.state = _broker_state(codex_pid=1234, sock_path=Path("/tmp/test-broker.sock"))

        with patch("codoxear.broker._proc_find_open_rollout_log", side_effect=RuntimeError("boom")):
            with patch.object(broker, "_teardown_managed_process_group") as teardown:
                broker._discover_log_watcher()

        teardown.assert_called_once_with()

    def test_sock_server_accept_failure_triggers_teardown(self) -> None:
        broker = Broker(cwd="/tmp", codex_args=[])
        with tempfile.TemporaryDirectory() as td:
            sock_root = Path(td)
            broker.state = _broker_state(codex_pid=1234, sock_path=sock_root / "broker.sock")
            with patch("codoxear.broker.SOCK_DIR", sock_root):
                with patch("codoxear.broker.socket.socket", return_value=_AcceptCrashSocket()):
                    with patch("codoxear.broker.os.chmod"):
                        with patch.object(broker, "_teardown_managed_process_group") as teardown:
                            broker._sock_server()

        teardown.assert_called_once_with()

    def test_web_owned_broker_forwards_local_stdin_when_tty_is_present(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: True, fileno=lambda: 9)
        _FakeThread.started_targets = []
        with tempfile.TemporaryDirectory() as td:
            broker = Broker(cwd=td, codex_args=[])
            broker.sessions_dir = Path(td) / "sessions"
            with patch("codoxear.broker.OWNER_TAG", "web"), patch("codoxear.broker.sys.stdin", fake_stdin), patch(
                "codoxear.broker._require_proc"
            ), patch("codoxear.broker._term_size", return_value=(24, 80)), patch(
                "codoxear.broker.pty.fork", return_value=(1234, 55)
            ), patch("codoxear.broker._set_winsize"), patch(
                "codoxear.broker.signal.signal"
            ), patch("codoxear.broker.termios.tcgetattr", return_value=["saved"]), patch(
                "codoxear.broker.termios.tcsetattr"
            ), patch("codoxear.broker.tty.setraw"), patch(
                "codoxear.broker.os.waitpid", return_value=(1234, 0)
            ), patch(
                "codoxear.broker.os.close"
            ), patch.object(
                broker, "_write_meta"
            ), patch(
                "codoxear.broker.threading.Thread", _FakeThread
            ):
                broker._emulate_terminal = False
                exit_code = broker.run()

        self.assertEqual(exit_code, 0)
        self.assertIn("_stdin_to_pty", _FakeThread.started_targets)

    def test_web_owned_headless_broker_keeps_local_stdin_disabled_without_tty(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        _FakeThread.started_targets = []
        with tempfile.TemporaryDirectory() as td:
            with patch("codoxear.broker.sys.stdin", fake_stdin):
                broker = Broker(cwd=td, codex_args=[])
            broker.sessions_dir = Path(td) / "sessions"
            with patch("codoxear.broker.OWNER_TAG", "web"), patch("codoxear.broker.sys.stdin", fake_stdin), patch(
                "codoxear.broker._require_proc"
            ), patch("codoxear.broker._term_size", return_value=(24, 80)), patch(
                "codoxear.broker.pty.fork", return_value=(1234, 55)
            ), patch("codoxear.broker._set_winsize"), patch(
                "codoxear.broker.signal.signal"
            ), patch(
                "codoxear.broker.os.waitpid", return_value=(1234, 0)
            ), patch(
                "codoxear.broker.os.close"
            ), patch.object(
                broker, "_write_meta"
            ), patch(
                "codoxear.broker.threading.Thread", _FakeThread
            ):
                exit_code = broker.run()

        self.assertEqual(exit_code, 0)
        self.assertNotIn("_stdin_to_pty", _FakeThread.started_targets)


if __name__ == "__main__":
    unittest.main()
