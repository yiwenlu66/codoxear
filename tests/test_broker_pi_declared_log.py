import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.broker import Broker, State


class TestBrokerPiDeclaredLog(unittest.TestCase):
    def test_discover_watcher_registers_declared_pi_log_when_file_appears(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sessions_dir = root / "sessions"
            session_dir = sessions_dir / "--tmp-project--"
            log_path = session_dir / "2026-04-07T00-00-00-000Z_test.jsonl"
            sock_path = root / "broker.sock"

            broker = Broker(cwd="/tmp/project", codex_args=[])
            broker._stop.clear()
            broker.state = State(
                codex_pid=123,
                pty_master_fd=0,
                cwd="/tmp/project",
                start_ts=0.0,
                codex_home=root,
                sessions_dir=sessions_dir,
                sock_path=sock_path,
                declared_log_path=log_path,
            )

            def fake_register(*, log_path: Path) -> None:
                with broker._lock:
                    assert broker.state is not None
                    broker.state.log_path = log_path
                broker._stop.set()

            def fake_sleep(_seconds: float) -> None:
                if not log_path.exists():
                    session_dir.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(
                        json.dumps({"type": "session", "id": "pi-session", "cwd": "/tmp/project"}) + "\n",
                        encoding="utf-8",
                    )

            with patch("codoxear.broker.AGENT_BACKEND", "pi"), \
                patch("codoxear.broker._proc_find_open_rollout_log", return_value=None), \
                patch("codoxear.broker._find_new_session_log", return_value=None), \
                patch("codoxear.broker._process_group_alive", return_value=True), \
                patch("codoxear.broker.os.waitpid", return_value=(0, 0)), \
                patch.object(Broker, "_maybe_register_or_switch_rollout", side_effect=fake_register) as register_mock, \
                patch("codoxear.broker.time.sleep", side_effect=fake_sleep):
                broker._discover_log_watcher()

            self.assertEqual(register_mock.call_count, 1)
            self.assertEqual(register_mock.call_args.kwargs["log_path"], log_path)
            self.assertIsNotNone(broker.state)
            self.assertEqual(broker.state.log_path, log_path)


if __name__ == "__main__":
    unittest.main()
