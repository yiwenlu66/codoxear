import json
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

    def test_list_sessions_uses_interrupted_idle_over_busy_log(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=False,
                interrupted_idle=True,
                queue_len=0,
            )
            mgr._sessions[s.session_id] = s
            mgr.idle_from_log_path = lambda _sid, _path: False  # type: ignore[method-assign]
            out = mgr.list_sessions()

        self.assertEqual(len(out), 1)
        self.assertIs(out[0].get("busy"), False)

    def test_interrupted_idle_does_not_override_unadvanced_confirmed_send(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=False,
                interrupted_idle=True,
                queue_len=0,
                last_send_boundary_active=True,
                last_send_log_path=log_path,
                last_send_log_size=log_path.stat().st_size,
            )
            mgr._sessions[s.session_id] = s
            mgr.idle_from_log_path = lambda _sid, _path: False  # type: ignore[method-assign]
            out = mgr.list_sessions()

            self.assertEqual(len(out), 1)
            self.assertIs(out[0].get("busy"), True)

            s.last_send_log_size -= 1
            out = mgr.list_sessions()

        self.assertEqual(len(out), 1)
        self.assertIs(out[0].get("busy"), False)

    def test_stale_idle_log_does_not_override_unadvanced_confirmed_send(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=False,
                interrupted_idle=True,
                queue_len=0,
                last_send_boundary_active=True,
                last_send_log_path=log_path,
                last_send_log_size=log_path.stat().st_size,
            )
            mgr._sessions[s.session_id] = s
            mgr.idle_from_log_path = lambda _sid, _path: True  # type: ignore[method-assign]
            out = mgr.list_sessions()

            self.assertEqual(len(out), 1)
            self.assertIs(out[0].get("busy"), True)

            s.last_send_log_size -= 1
            out = mgr.list_sessions()

        self.assertEqual(len(out), 1)
        self.assertIs(out[0].get("busy"), False)

    def test_missing_log_does_not_override_unadvanced_confirmed_send(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "missing.jsonl"
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=False,
                interrupted_idle=True,
                queue_len=0,
                last_send_boundary_active=True,
                last_send_log_path=log_path,
                last_send_log_size=10,
            )
            mgr._sessions[s.session_id] = s
            mgr.idle_from_log_path = lambda _sid, _path: self.fail("missing log must not be parsed for list display")  # type: ignore[method-assign]
            out = mgr.list_sessions()

            self.assertEqual(len(out), 1)
            self.assertIs(out[0].get("busy"), True)

            s.last_send_log_path = Path(td) / "other.jsonl"
            out = mgr.list_sessions()

        self.assertEqual(len(out), 1)
        self.assertIs(out[0].get("busy"), False)

    def test_list_sessions_skips_non_object_jsonl_rows(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('[]\n{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=False,
                queue_len=0,
            )
            mgr._sessions[s.session_id] = s
            mgr.idle_from_log_path = lambda _sid, _path: True  # type: ignore[method-assign]

            out = mgr.list_sessions()

        self.assertEqual(len(out), 1)
        self.assertIs(out[0].get("busy"), False)

    def test_no_log_confirmed_send_boundary_keeps_list_busy_until_nonempty_log(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=None,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=False,
                interrupted_idle=True,
                queue_len=0,
                last_send_boundary_active=True,
                last_send_log_path=None,
                last_send_log_size=None,
            )
            mgr._sessions[s.session_id] = s
            out = mgr.list_sessions()

            self.assertEqual(len(out), 1)
            self.assertIs(out[0].get("busy"), True)

            log_path.write_text("", encoding="utf-8")
            s.log_path = log_path
            mgr.idle_from_log_path = lambda _sid, _path: True  # type: ignore[method-assign]
            out = mgr.list_sessions()

            self.assertEqual(len(out), 1)
            self.assertIs(out[0].get("busy"), True)
            self.assertTrue(s.last_send_boundary_active)

            for content in ("\n", "garbage\n", "[]\n", '{"type":"message","message":{"role":"assistant","content":['):
                log_path.write_text(content, encoding="utf-8")
                out = mgr.list_sessions()

                self.assertEqual(len(out), 1)
                self.assertIs(out[0].get("busy"), True)
                self.assertTrue(s.last_send_boundary_active)

            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            out = mgr.list_sessions()

            self.assertEqual(len(out), 1)
            self.assertIs(out[0].get("busy"), False)
            self.assertFalse(s.last_send_boundary_active)
            self.assertIsNone(s.last_send_log_path)
            self.assertIsNone(s.last_send_log_size)

            s.log_path = None
            out = mgr.list_sessions()

        self.assertEqual(len(out), 1)
        self.assertIs(out[0].get("busy"), False)

    def test_interrupted_idle_does_not_override_nonempty_broker_queue(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=False,
                interrupted_idle=True,
                queue_len=1,
            )
            mgr._sessions[s.session_id] = s
            mgr.idle_from_log_path = lambda _sid, _path: False  # type: ignore[method-assign]
            out = mgr.list_sessions()

        self.assertEqual(len(out), 1)
        self.assertIs(out[0].get("busy"), True)

    def test_interrupted_idle_does_not_override_busy_broker(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=True,
                interrupted_idle=True,
                queue_len=0,
            )
            mgr._sessions[s.session_id] = s
            mgr.idle_from_log_path = lambda _sid, _path: False  # type: ignore[method-assign]
            out = mgr.list_sessions()

        self.assertEqual(len(out), 1)
        self.assertIs(out[0].get("busy"), True)

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

    def test_message_snapshot_uses_interrupted_idle_over_busy_log(self) -> None:
        class _Manager:
            def get_state(self, _session_id):
                return {"busy": False, "queue_len": 0, "interrupted_idle": True}

            def idle_from_log(self, _session_id):
                return False

            def _queue_len(self, _session_id):
                return 0

        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
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

    def test_message_snapshot_rejects_interrupted_idle_before_confirmed_send_advances(self) -> None:
        class _Manager:
            def get_state(self, _session_id):
                return {"busy": False, "queue_len": 0, "interrupted_idle": True}

            def idle_from_log(self, _session_id):
                return False

            def _queue_len(self, _session_id):
                return 0

        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=True,
                queue_len=0,
                last_send_boundary_active=True,
                last_send_log_path=log_path,
                last_send_log_size=log_path.stat().st_size,
            )
            with patch("codoxear.server.MANAGER", _Manager()):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

            self.assertIs(busy, True)
            self.assertEqual(queue_len, 0)

            s.last_send_log_size -= 1
            with patch("codoxear.server.MANAGER", _Manager()):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_rejects_stale_idle_log_before_confirmed_send_advances(self) -> None:
        class _Manager:
            def get_state(self, _session_id):
                return {"busy": False, "queue_len": 0, "interrupted_idle": True}

            def idle_from_log(self, _session_id):
                return True

            def _queue_len(self, _session_id):
                return 0

        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=True,
                queue_len=0,
                last_send_boundary_active=True,
                last_send_log_path=log_path,
                last_send_log_size=log_path.stat().st_size,
            )
            with patch("codoxear.server.MANAGER", _Manager()):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

            self.assertIs(busy, True)
            self.assertEqual(queue_len, 0)

            s.last_send_log_size -= 1
            with patch("codoxear.server.MANAGER", _Manager()):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_rejects_missing_log_before_confirmed_send_advances(self) -> None:
        class _Manager:
            def get_state(self, _session_id):
                return {"busy": False, "queue_len": 0, "interrupted_idle": True}

            def idle_from_log(self, _session_id):
                raise AssertionError("missing log must not be parsed for message snapshot")

            def _queue_len(self, _session_id):
                return 0

        with TemporaryDirectory() as td:
            log_path = Path(td) / "missing.jsonl"
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
                busy=True,
                queue_len=0,
                last_send_boundary_active=True,
                last_send_log_path=log_path,
                last_send_log_size=10,
            )
            with patch("codoxear.server.MANAGER", _Manager()):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

            self.assertIs(busy, True)
            self.assertEqual(queue_len, 0)

            s.last_send_log_path = Path(td) / "other.jsonl"
            with patch("codoxear.server.MANAGER", _Manager()):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_rejects_no_log_confirmed_send_boundary_until_nonempty_log(self) -> None:
        mgr = _make_manager()
        s = Session(
            session_id="broker-1",
            thread_id="broker-1",
            broker_pid=1,
            codex_pid=2,
            agent_backend="pi",
            owned=False,
            start_ts=123.0,
            cwd="/tmp",
            log_path=None,
            sock_path=Path("/tmp/broker-1.sock"),
            busy=True,
            queue_len=0,
            last_send_boundary_active=True,
            last_send_log_path=None,
            last_send_log_size=None,
        )
        mgr._sessions[s.session_id] = s
        mgr.get_state = lambda _sid: {"busy": False, "queue_len": 0, "interrupted_idle": True}  # type: ignore[method-assign]
        mgr._queue_len = lambda _sid: 0  # type: ignore[method-assign]
        mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

        with patch("codoxear.server.MANAGER", mgr):
            _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

        self.assertIs(busy, True)
        self.assertEqual(queue_len, 0)
        self.assertTrue(s.last_send_boundary_active)

        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text("", encoding="utf-8")
            s.log_path = log_path
            with patch("codoxear.server.MANAGER", mgr):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

            self.assertIs(busy, True)
            self.assertEqual(queue_len, 0)
            self.assertTrue(s.last_send_boundary_active)

            for content in ("\n", "garbage\n", "[]\n", '{"type":"message","message":{"role":"assistant","content":['):
                log_path.write_text(content, encoding="utf-8")
                with patch("codoxear.server.MANAGER", mgr):
                    _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

                self.assertIs(busy, True)
                self.assertEqual(queue_len, 0)
                self.assertTrue(s.last_send_boundary_active)

            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            with patch("codoxear.server.MANAGER", mgr):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

            self.assertIs(busy, False)
            self.assertEqual(queue_len, 0)
            self.assertFalse(s.last_send_boundary_active)
            self.assertIsNone(s.last_send_log_path)
            self.assertIsNone(s.last_send_log_size)

            s.log_path = None
            with patch("codoxear.server.MANAGER", mgr):
                _state, busy, queue_len, _token = _message_runtime_snapshot("broker-1", s)

        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_rejects_malformed_interrupted_idle(self) -> None:
        class _Manager:
            def get_state(self, _session_id):
                return {"busy": False, "queue_len": 0, "interrupted_idle": "true"}

            def idle_from_log(self, _session_id):
                return False

            def _queue_len(self, _session_id):
                return 0

        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=log_path,
                sock_path=Path("/tmp/broker-1.sock"),
            )
            with patch("codoxear.server.MANAGER", _Manager()):
                with self.assertRaisesRegex(ValueError, "invalid broker state response"):
                    _message_runtime_snapshot("broker-1", s)

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

    def test_refresh_session_meta_clears_interrupted_idle_on_log_change(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td:
            root = Path(td)
            old_log = root / "old.jsonl"
            new_log = root / "new.jsonl"
            old_log.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"old"}]}}\n', encoding="utf-8")
            new_log.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"new"}]}}\n', encoding="utf-8")
            sock = root / "broker.sock"
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "cwd": "/tmp",
                        "log_path": str(new_log),
                        "agent_backend": "pi",
                        "codex_pid": 2,
                        "broker_pid": 1,
                        "start_ts": 123.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            s = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="pi",
                owned=False,
                start_ts=123.0,
                cwd="/tmp",
                log_path=old_log,
                sock_path=sock,
                interrupted_idle=True,
            )
            mgr._sessions[s.session_id] = s

            SessionManager.refresh_session_meta(mgr, s.session_id, drain_queue=False)

        self.assertEqual(s.log_path, new_log)
        self.assertFalse(s.interrupted_idle)

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

    def test_get_state_refreshes_interrupted_idle_cache(self) -> None:
        mgr = _make_manager()
        s = Session(
            session_id="broker-1",
            thread_id="broker-1",
            broker_pid=1,
            codex_pid=2,
            agent_backend="pi",
            owned=False,
            start_ts=123.0,
            cwd="/tmp",
            log_path=None,
            sock_path=Path("/tmp/broker-1.sock"),
            busy=False,
            interrupted_idle=True,
            queue_len=0,
        )
        mgr._sessions[s.session_id] = s
        mgr._sock_call = lambda *_args, **_kwargs: {"busy": False, "queue_len": 0, "interrupted_idle": False}  # type: ignore[method-assign]

        state = mgr.get_state(s.session_id)

        self.assertIs(state["interrupted_idle"], False)
        self.assertFalse(s.interrupted_idle)


if __name__ == "__main__":
    unittest.main()
