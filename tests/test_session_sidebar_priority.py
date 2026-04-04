import threading
import time
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from codoxear.server import Session
from codoxear.server import SessionManager


def _make_manager() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._sessions = {}
    mgr._harness = {}
    mgr._aliases = {}
    mgr._sidebar_meta = {}
    mgr._hidden_sessions = set()
    mgr._files = {}
    mgr._queues = {}
    mgr._recent_cwds = {}
    mgr._discover_existing_if_stale = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._prune_dead_sessions = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._update_meta_counters = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_sidebar_meta = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_hidden_sessions = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_aliases = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_harness = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_files = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_queues = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_recent_cwds = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return mgr


def _session(*, sid: str, start_ts: float, last_chat_ts: float | None = None, owned: bool = False) -> Session:
    return Session(
        session_id=sid,
        thread_id=sid,
        broker_pid=100,
        codex_pid=200,
        agent_backend="codex",
        owned=owned,
        start_ts=start_ts,
        cwd=f"/tmp/{sid}",
        log_path=None,
        sock_path=Path(f"/tmp/{sid}.sock"),
        last_chat_ts=last_chat_ts,
    )


class TestSessionSidebarPriority(unittest.TestCase):
    def test_list_sessions_sorts_by_final_priority_then_recency(self) -> None:
        mgr = _make_manager()
        now = time.time()
        recent = _session(sid="recent", start_ts=now - 100, last_chat_ts=now - 300)
        older = _session(sid="older", start_ts=now - 200, last_chat_ts=now - 16 * 3600)
        mgr._sessions = {recent.session_id: recent, older.session_id: older}
        mgr._sidebar_meta = {
            "recent": {"priority_offset": -0.8},
            "older": {"priority_offset": 0.2},
        }

        rows = mgr.list_sessions()

        self.assertEqual([row["session_id"] for row in rows], ["older", "recent"])
        self.assertAlmostEqual(rows[0]["final_priority"], 0.45, delta=0.04)
        self.assertAlmostEqual(rows[1]["final_priority"], 0.19, delta=0.04)

    def test_list_sessions_clears_expired_snooze_and_stale_dependency(self) -> None:
        mgr = _make_manager()
        now = time.time()
        current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 50)
        mgr._sessions = {current.session_id: current}
        mgr._sidebar_meta = {
            "current": {
                "priority_offset": 0.1,
                "snooze_until": now - 5,
                "dependency_session_id": "missing",
            }
        }

        rows = mgr.list_sessions()

        self.assertEqual(rows[0]["session_id"], "current")
        self.assertFalse(rows[0]["snoozed"])
        self.assertFalse(rows[0]["blocked"])
        self.assertIsNone(rows[0]["snooze_until"])
        self.assertIsNone(rows[0]["dependency_session_id"])

    def test_delete_session_kills_terminal_owned_and_clears_dependents(self) -> None:
        mgr = _make_manager()
        now = time.time()
        blocked = _session(sid="blocked", start_ts=now - 100, last_chat_ts=now - 10)
        target = _session(sid="target", start_ts=now - 200, last_chat_ts=now - 20, owned=False)
        mgr._sessions = {blocked.session_id: blocked, target.session_id: target}
        mgr._sidebar_meta = {
            "blocked": {"priority_offset": 0.0, "dependency_session_id": "target"},
            "target": {"priority_offset": 0.5},
        }
        mgr._queues = {"target": ["queued"]}
        mgr._harness = {"target": {"enabled": True, "request": "x"}}
        mgr._files = {"sid:target": ["/tmp/target/a.py"]}
        called = {"shutdown": 0}

        def _sock_call(*args, **kwargs):
            called["shutdown"] += 1
            return {"ok": True}

        mgr._sock_call = _sock_call  # type: ignore[method-assign]

        ok = mgr.delete_session("target")

        self.assertTrue(ok)
        self.assertEqual(called["shutdown"], 1)
        self.assertNotIn("target", mgr._sidebar_meta)
        self.assertNotIn("target", mgr._queues)
        self.assertNotIn("target", mgr._harness)
        self.assertNotIn("sid:target", mgr._files)
        self.assertIsNone(mgr._sidebar_meta["blocked"].get("dependency_session_id"))
        self.assertIn("target", mgr._hidden_sessions)

    def test_kill_session_falls_back_to_pid_teardown_when_socket_is_dead(self) -> None:
        mgr = _make_manager()
        s = _session(sid="target", start_ts=time.time() - 10, last_chat_ts=None, owned=False)
        mgr._sessions = {s.session_id: s}

        def _sock_call(*args, **kwargs):
            raise OSError("dead socket")

        mgr._sock_call = _sock_call  # type: ignore[method-assign]

        with patch.object(mgr, "_kill_session_via_pids", return_value=True) as kill_via_pids:
            ok = mgr.kill_session("target")

        self.assertTrue(ok)
        kill_via_pids.assert_called_once_with(s)

    def test_kill_session_via_pids_prunes_stale_metadata_without_signals(self) -> None:
        mgr = _make_manager()
        s = _session(sid="target", start_ts=time.time() - 10, last_chat_ts=None, owned=False)

        with patch("codoxear.server._process_group_alive", return_value=False):
            with patch("codoxear.server._pid_alive", return_value=False):
                with patch("codoxear.server._terminate_process_group") as kill_group:
                    with patch("codoxear.server._terminate_process") as kill_proc:
                        with patch("codoxear.server._unlink_quiet") as unlink:
                            ok = mgr._kill_session_via_pids(s)

        self.assertTrue(ok)
        kill_group.assert_not_called()
        kill_proc.assert_not_called()
        self.assertEqual(unlink.call_count, 2)

    def test_edit_session_is_atomic_when_dependency_invalid(self) -> None:
        mgr = _make_manager()
        now = time.time()
        s = _session(sid="edit", start_ts=now - 100, last_chat_ts=now - 20)
        mgr._sessions = {s.session_id: s}
        mgr._aliases = {"edit": "old name"}
        mgr._sidebar_meta = {"edit": {"priority_offset": 0.1}}

        with self.assertRaisesRegex(ValueError, "dependency session not found"):
            mgr.edit_session(
                "edit",
                name="new name",
                priority_offset=0.2,
                snooze_until=None,
                dependency_session_id="missing",
            )

        self.assertEqual(mgr._aliases["edit"], "old name")
        self.assertEqual(mgr._sidebar_meta["edit"]["priority_offset"], 0.1)

    def test_list_sessions_uses_start_ts_when_log_has_no_sidebar_relevant_message(self) -> None:
        mgr = _make_manager()
        mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]
        with unittest.mock.patch("codoxear.server._last_conversation_ts_from_tail", return_value=None):
            s = _session(sid="nologmsg", start_ts=123.0, last_chat_ts=None)
            s.log_path = Path("/tmp/fake-rollout.jsonl")
            mgr._sessions = {s.session_id: s}
            original_exists = Path.exists

            def _exists(path_obj):
                if str(path_obj) == "/tmp/fake-rollout.jsonl":
                    return True
                return original_exists(path_obj)

            with unittest.mock.patch("pathlib.Path.exists", _exists):
                rows = mgr.list_sessions()

        self.assertEqual(rows[0]["updated_ts"], 123.0)

    def test_list_sessions_backfills_updated_ts_from_large_preexisting_log(self) -> None:
        mgr = _make_manager()
        mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "rollout-2026-03-17T00-00-00-eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee.jsonl"
            now = time.time()
            user_ts = now - 30
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "session_meta", "payload": {"id": "current", "source": "cli"}}),
                        json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "real turn"}, "ts": user_ts}),
                        json.dumps(
                            {
                                "type": "response_item",
                                "payload": {
                                    "type": "function_call",
                                    "name": "tool",
                                    "arguments": {"blob": "x" * (400 * 1024)},
                                },
                                "ts": now - 20,
                            }
                        ),
                        json.dumps({"type": "response_item", "payload": {"type": "reasoning"}, "ts": now - 10}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            current = _session(sid="current", start_ts=10.0, last_chat_ts=None)
            current.log_path = log_path
            mgr._sessions = {current.session_id: current}

            rows = mgr.list_sessions()

        self.assertAlmostEqual(rows[0]["updated_ts"], user_ts, places=3)

    def test_list_sessions_scans_preexisting_history_only_once(self) -> None:
        mgr = _make_manager()
        mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "rollout-2026-03-17T00-00-00-ffffffff-ffff-ffff-ffff-ffffffffffff.jsonl"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "session_meta", "payload": {"id": "current", "source": "cli"}}),
                        json.dumps({"type": "event_msg", "payload": {"type": "agent_reasoning"}, "ts": time.time()}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            current = _session(sid="current", start_ts=123.0, last_chat_ts=None)
            current.log_path = log_path
            mgr._sessions = {current.session_id: current}

            with patch("codoxear.server._last_conversation_ts_from_tail", return_value=None) as backfill:
                rows1 = mgr.list_sessions()
                rows2 = mgr.list_sessions()

        self.assertEqual(backfill.call_count, 1)
        self.assertEqual(rows1[0]["updated_ts"], 123.0)
        self.assertEqual(rows2[0]["updated_ts"], 123.0)

    def test_recent_cwds_include_backfilled_history_and_live_sessions(self) -> None:
        mgr = _make_manager()
        mgr._recent_cwds = {"/repo/ended": 100.0}
        now = time.time()
        current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 5)
        mgr._sessions = {current.session_id: current}
        mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

        mgr.list_sessions()

        self.assertEqual(mgr.recent_cwds(limit=4), ["/tmp/current", "/repo/ended"])

    def test_list_sessions_exposes_model_and_reasoning_effort(self) -> None:
        mgr = _make_manager()
        now = time.time()
        current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 5)
        current.model = "gpt-5.4"
        current.reasoning_effort = "xhigh"
        mgr._sessions = {current.session_id: current}
        mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

        rows = mgr.list_sessions()

        self.assertEqual(rows[0]["model"], "gpt-5.4")
        self.assertEqual(rows[0]["reasoning_effort"], "xhigh")

    def test_list_sessions_exposes_tmux_transport(self) -> None:
        mgr = _make_manager()
        now = time.time()
        current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 5)
        current.transport = "tmux"
        current.tmux_session = "codoxear"
        current.tmux_window = "current-abcd12"
        mgr._sessions = {current.session_id: current}
        mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

        rows = mgr.list_sessions()

        self.assertEqual(rows[0]["transport"], "tmux")
        self.assertEqual(rows[0]["tmux_session"], "codoxear")
        self.assertEqual(rows[0]["tmux_window"], "current-abcd12")

    def test_list_sessions_falls_back_to_log_run_settings(self) -> None:
        mgr = _make_manager()
        now = time.time()
        current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 5)
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "rollout-2026-03-17T00-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            log_path.write_text(
                "\n".join(
                    [
                        '{"type":"session_meta","payload":{"id":"current","cwd":"/tmp/current","timestamp":"2026-03-17T00:00:00Z"}}',
                        '{"type":"turn_context","payload":{"model":"gpt-5.4","reasoning_effort":"high"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            current.log_path = log_path
            mgr._sessions = {current.session_id: current}
            mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]

            rows = mgr.list_sessions()

        self.assertEqual(rows[0]["model"], "gpt-5.4")
        self.assertEqual(rows[0]["reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
