import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codoxear.sessiond import Sessiond
from codoxear.sessiond import State


def _sessiond_state(*, codex_pid: int, sock_path: Path) -> State:
    return State(
        session_id="sid",
        codex_pid=codex_pid,
        log_path=Path("/tmp/pending.jsonl"),
        sock_path=sock_path,
        pty_master_fd=1,
        start_ts=0.0,
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


class TestSessiondFailClosed(unittest.TestCase):
    def test_teardown_managed_process_group_kills_real_process_group(self) -> None:
        proc = subprocess.Popen(["sh", "-c", "sleep 100"], start_new_session=True)
        try:
            sessiond = Sessiond(cwd="/tmp", codex_args=[])
            sessiond.state = _sessiond_state(codex_pid=proc.pid, sock_path=Path("/tmp/test-sessiond.sock"))

            sessiond._teardown_managed_process_group(wait_seconds=0.2)

            proc.wait(timeout=2.0)
            self.assertIsNotNone(proc.returncode)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)

    def test_sock_server_accept_failure_triggers_teardown(self) -> None:
        sessiond = Sessiond(cwd="/tmp", codex_args=[])
        with tempfile.TemporaryDirectory() as td:
            sessiond.state = _sessiond_state(codex_pid=1234, sock_path=Path(td) / "sessiond.sock")
            with patch("codoxear.sessiond.socket.socket", return_value=_AcceptCrashSocket()):
                with patch("codoxear.sessiond.os.chmod"):
                    with patch.object(sessiond, "_teardown_managed_process_group") as teardown:
                        sessiond._sock_server()

        teardown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
