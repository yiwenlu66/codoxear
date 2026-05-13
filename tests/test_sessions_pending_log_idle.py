import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.server import Session
from codoxear.server import SessionManager
from codoxear.server import _message_runtime_snapshot


def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._sessions = {}
    mgr._harness = {}
    mgr._aliases = {}
    mgr._files = {}
    mgr._discover_existing_if_stale = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._prune_dead_sessions = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._update_meta_counters = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return mgr


class TestSessionsPendingLogIdle(unittest.TestCase):
    def test_list_sessions_forces_idle_when_log_is_none(self) -> None:
        mgr = _make_manager()
        s = Session(
            session_id="broker-1",
            thread_id="broker-1",
            broker_pid=1,
            codex_pid=2,
            agent_backend="codex",
            owned=False,
            start_ts=123.0,
            cwd="/tmp",
            log_path=None,
            sock_path=Path("/tmp/broker-1.sock"),
            busy=True,
            queue_len=0,
        )
        mgr._sessions[s.session_id] = s

        out = mgr.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0].get("busy"), False)

    def test_list_sessions_uses_log_idle_over_stale_broker_busy(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"broker-1","source":"cli"}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="codex",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=True,
                queue_len=0,
            )
            mgr._sessions[s.session_id] = s
            mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]
            out = mgr.list_sessions()

        self.assertEqual(len(out), 1)
        self.assertIs(out[0].get("busy"), False)

    def test_message_snapshot_uses_log_idle_over_stale_broker_busy(self) -> None:
        class _Manager:
            def get_state(self, _session_id):
                return {"busy": True, "queue_len": 0}

            def idle_from_log(self, _session_id):
                return True

            def _queue_len(self, _session_id):
                return 0

        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"broker-1","source":"cli"}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="codex",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=True,
                queue_len=0,
            )
            with unittest.mock.patch("codoxear.server.MANAGER", _Manager()):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)


if __name__ == "__main__":
    unittest.main()
