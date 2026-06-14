import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.server import Session
from codoxear.server import SessionManager
from codoxear.server import _message_runtime_snapshot


def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._sessions = {}
    mgr._unattended = {}
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

    def test_idle_from_log_path_survives_detach_after_row_snapshot(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"thread-old","source":"cli"}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="thread-new",
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

            self.assertIs(mgr.idle_from_log_path(s.session_id, log_path), True)

        self.assertEqual(s.idle_cache_log_off, -1)
        self.assertIsNone(s.idle_cache_value)

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
            with patch("codoxear.server.MANAGER", _Manager()):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_rejects_malformed_mocked_broker_state(self) -> None:
        class _Manager:
            def get_state(self, _session_id):
                return {"busy": "false", "queue_len": 0}

            def _queue_len(self, _session_id):
                return 0

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
        )
        with patch("codoxear.server.MANAGER", _Manager()):
            with self.assertRaisesRegex(ValueError, "invalid broker state response"):
                _message_runtime_snapshot("broker-1", s)

    def test_message_snapshot_prefers_log_token_over_stale_broker_token(self) -> None:
        class _Manager:
            def get_state(self, _session_id):
                return {"busy": False, "queue_len": 0, "token": {"tokens_in_context": 0}}

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
                busy=False,
                queue_len=0,
                token={"tokens_in_context": 185136},
            )
            with patch("codoxear.server.MANAGER", _Manager()):
                _state, _busy, _queue_len, token = _message_runtime_snapshot("broker-1", s)

        self.assertEqual(token, {"tokens_in_context": 185136})

    def test_message_snapshot_ignores_stale_broker_token_when_log_has_no_token_yet(self) -> None:
        class _Manager:
            def get_state(self, _session_id):
                return {"busy": False, "queue_len": 0, "token": {"tokens_in_context": 0}}

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
                busy=False,
                queue_len=0,
                token=None,
            )
            with patch("codoxear.server.MANAGER", _Manager()):
                _state, _busy, _queue_len, token = _message_runtime_snapshot("broker-1", s)

        self.assertIsNone(token)

    def test_refresh_session_state_does_not_overwrite_log_token_with_stale_broker_token(self) -> None:
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
                busy=False,
                queue_len=0,
                token={"tokens_in_context": 185136},
            )
            mgr._sessions[s.session_id] = s
            mgr._sock_call = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
                "busy": False,
                "queue_len": 0,
                "token": {"tokens_in_context": 0},
            }

            ok, err = mgr._refresh_session_state(s.session_id, s.sock_path)

        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertEqual(s.token, {"tokens_in_context": 185136})

    def test_refresh_session_state_rejects_malformed_broker_state_without_coercion(self) -> None:
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
            busy=False,
            queue_len=0,
        )
        mgr._sessions[s.session_id] = s
        malformed = [
            {"busy": "false", "queue_len": 0},
            {"busy": False, "queue_len": "0"},
            {"busy": False, "queue_len": -1},
            {"busy": False, "queue_len": True},
        ]
        for state in malformed:
            with self.subTest(state=state):
                mgr._sock_call = lambda *_args, state=state, **_kwargs: state  # type: ignore[method-assign]
                ok, err = mgr._refresh_session_state(s.session_id, s.sock_path)
                self.assertFalse(ok)
                self.assertIsInstance(err, ValueError)
                self.assertFalse(s.busy)
                self.assertEqual(s.queue_len, 0)

    def test_get_state_does_not_overwrite_log_token_with_stale_broker_token(self) -> None:
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
                busy=False,
                queue_len=0,
                token={"tokens_in_context": 185136},
            )
            mgr._sessions[s.session_id] = s
            mgr._sock_call = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
                "busy": False,
                "queue_len": 0,
                "token": {"tokens_in_context": 0},
            }

            state = mgr.get_state(s.session_id)

        self.assertEqual(state["token"], {"tokens_in_context": 0})
        self.assertEqual(s.token, {"tokens_in_context": 185136})


if __name__ == "__main__":
    unittest.main()
