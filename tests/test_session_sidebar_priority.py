import threading
import time
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from codoxear.page_state_sqlite import DurableSessionRecord
from codoxear.page_state_sqlite import PageStateDB
from codoxear.server import Session
from codoxear.server import SessionManager
from codoxear.server import _frontend_session_list_row
from codoxear.server import _session_list_payload


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
    mgr._cwd_groups = {}
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
    mgr._save_cwd_groups = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._include_historical_sessions = False
    return mgr


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(obj) + "\n" for obj in objs), encoding="utf-8")


def _session(
    *, sid: str, start_ts: float, last_chat_ts: float | None = None, owned: bool = False
) -> Session:
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
        self.assertEqual(rows[0]["runtime_id"], "current")
        self.assertFalse(rows[0]["focused"])
        self.assertFalse(rows[0]["snoozed"])
        self.assertFalse(rows[0]["blocked"])
        self.assertIsNone(rows[0]["snooze_until"])
        self.assertIsNone(rows[0]["dependency_session_id"])

    def test_frontend_session_row_prefers_title_over_first_user_message(self) -> None:
        row = _frontend_session_list_row(
            {
                "session_id": "sess-1",
                "title": "Structured title",
                "first_user_message": "draft prompt",
            }
        )

        self.assertEqual(row["display_name"], "Structured title")

    def test_sidebar_meta_get_defaults_focus_to_false(self) -> None:
        mgr = _make_manager()
        mgr._page_state_ref_for_session_id = lambda session_id: ("pi", session_id)  # type: ignore[method-assign]

        meta = mgr.sidebar_meta_get("sess-1")

        self.assertFalse(meta["focused"])

    def test_session_list_payload_keeps_focused_rows_visible(self) -> None:
        payload = _session_list_payload(
            [
                {"session_id": f"sess-{idx}", "cwd": "/tmp/project", "updated_ts": 100 - idx, "focused": idx == 6}
                for idx in range(7)
            ]
        )

        session_ids = [row["session_id"] for row in payload["sessions"]]
        self.assertIn("sess-6", session_ids)

    def test_list_sessions_includes_pending_sqlite_sessions_as_live_placeholders(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr._refresh_durable_session_catalog = lambda *args, **kwargs: None  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            db = PageStateDB(Path(td) / "state.sqlite")
            db.save_sessions(
                {
                    ("pi", "resume-pi"): DurableSessionRecord(
                        backend="pi",
                        session_id="resume-pi",
                        cwd="/repo",
                        source_path=str(Path(td) / "resume-pi.jsonl"),
                        title="Recovered",
                        first_user_message="hello",
                        created_at=100.0,
                        updated_at=150.0,
                        pending_startup=True,
                    )
                }
            )
            db.save_session_ui_state(
                aliases={("pi", "resume-pi"): "Recovered alias"},
                sidebar_meta={("pi", "resume-pi"): {"priority_offset": 0.2, "focused": True}},
                hidden_keys=set(),
            )
            mgr._aliases, mgr._sidebar_meta, mgr._hidden_sessions = db.load_session_ui_state()
            mgr._page_state_db = db
            rows = mgr.list_sessions()
            db.close()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session_id"], "resume-pi")
        self.assertIsNone(rows[0]["runtime_id"])
        self.assertFalse(rows[0]["historical"])
        self.assertTrue(rows[0]["pending_startup"])
        self.assertTrue(rows[0]["focused"])
        self.assertEqual(rows[0]["alias"], "Recovered alias")
        self.assertEqual(rows[0]["title"], "Recovered")
        self.assertEqual(rows[0]["display_name"], "Recovered alias")

    def test_list_sessions_includes_recovered_sqlite_sessions_when_no_live_sessions(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr._refresh_durable_session_catalog = lambda *args, **kwargs: None  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            db = PageStateDB(Path(td) / "state.sqlite")
            db.save_sessions(
                {
                    ("pi", "resume-pi"): DurableSessionRecord(
                        backend="pi",
                        session_id="resume-pi",
                        cwd="/repo",
                        source_path=str(Path(td) / "resume-pi.jsonl"),
                        title="Recovered",
                        first_user_message="hello",
                        created_at=100.0,
                        updated_at=150.0,
                    )
                }
            )
            db.save_session_ui_state(
                aliases={("pi", "resume-pi"): "Recovered alias"},
                sidebar_meta={("pi", "resume-pi"): {"priority_offset": 0.2, "focused": True}},
                hidden_keys=set(),
            )
            mgr._aliases, mgr._sidebar_meta, mgr._hidden_sessions = db.load_session_ui_state()
            mgr._page_state_db = db
            rows = mgr.list_sessions()
            db.close()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session_id"], "history:pi:resume-pi")
        self.assertEqual(rows[0]["runtime_id"], None)
        self.assertTrue(rows[0]["historical"])
        self.assertTrue(rows[0]["focused"])
        self.assertEqual(rows[0]["alias"], "Recovered alias")
        self.assertEqual(rows[0]["title"], "Recovered")
        self.assertEqual(rows[0]["first_user_message"], "hello")
        self.assertEqual(rows[0]["display_name"], "Recovered alias")

    def test_list_sessions_includes_historical_resumable_sessions_when_no_live_sessions(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr._include_historical_sessions = True
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            codex_log = root / "rollout-2026-04-10T01-00-00-history-a.jsonl"
            pi_root = root / "pi-native"
            pi_session = pi_root / "--repo--" / "2026-04-10T01-00-00-000Z_pi.jsonl"
            _write_jsonl(
                codex_log,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "resume-codex",
                            "cwd": "/repo",
                            "timestamp": "2026-04-10T01:00:00Z",
                            "source": "cli",
                        },
                    }
                ],
            )
            _write_jsonl(
                pi_session,
                [
                    {
                        "type": "session",
                        "id": "resume-pi",
                        "cwd": "/repo",
                        "timestamp": "2026-04-10T01:00:05Z",
                    }
                ],
            )

            with (
                patch("codoxear.server._iter_session_logs", return_value=[codex_log]),
                patch("codoxear.server.PI_NATIVE_SESSIONS_DIR", pi_root),
            ):
                rows = mgr.list_sessions()

        self.assertEqual(
            [(row["agent_backend"], row.get("resume_session_id")) for row in rows],
            [("pi", "resume-pi"), ("codex", "resume-codex")],
        )
        self.assertTrue(all(row.get("historical") is True for row in rows))
        self.assertTrue(
            all(str(row["session_id"]).startswith("history:") for row in rows)
        )
        self.assertTrue(all(row.get("runtime_id") is None for row in rows))
        self.assertEqual({row["cwd"] for row in rows}, {"/repo"})

    def test_list_sessions_skips_historical_entry_when_matching_live_session_exists(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr._include_historical_sessions = True
        live = Session(
            session_id="live-broker",
            thread_id="resume-codex",
            broker_pid=100,
            codex_pid=200,
            agent_backend="codex",
            owned=False,
            start_ts=time.time() - 20,
            cwd="/repo",
            log_path=None,
            sock_path=Path("/tmp/live-broker.sock"),
        )
        mgr._sessions = {live.session_id: live}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            live_log = root / "rollout-2026-04-10T01-00-00-live.jsonl"
            other_log = root / "rollout-2026-04-10T01-00-01-other.jsonl"
            pi_root = root / "pi-native"
            _write_jsonl(
                live_log,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "resume-codex",
                            "cwd": "/repo",
                            "timestamp": "2026-04-10T01:00:00Z",
                            "source": "cli",
                        },
                    }
                ],
            )
            _write_jsonl(
                other_log,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "resume-other",
                            "cwd": "/repo",
                            "timestamp": "2026-04-10T01:00:01Z",
                            "source": "cli",
                        },
                    }
                ],
            )

            with (
                patch(
                    "codoxear.server._iter_session_logs",
                    return_value=[other_log, live_log],
                ),
                patch("codoxear.server.PI_NATIVE_SESSIONS_DIR", pi_root),
            ):
                rows = mgr.list_sessions()

        self.assertEqual(
            [row["session_id"] for row in rows],
            ["history:codex:resume-other", "resume-codex"],
        )
        self.assertEqual(rows[0]["resume_session_id"], "resume-other")
        self.assertTrue(rows[0]["historical"])
        self.assertFalse(rows[1].get("historical", False))

    def test_list_sessions_dedupes_live_rows_for_same_backend_thread(self) -> None:
        mgr = _make_manager()
        now = time.time()
        newest = Session(
            session_id="broker-new",
            thread_id="shared-thread",
            broker_pid=100,
            codex_pid=200,
            agent_backend="pi",
            backend="pi",
            owned=True,
            start_ts=now - 20,
            cwd="/tmp/repo",
            log_path=None,
            sock_path=Path("/tmp/broker-new.sock"),
            last_chat_ts=now - 5,
        )
        duplicate = Session(
            session_id="broker-old",
            thread_id="shared-thread",
            broker_pid=101,
            codex_pid=201,
            agent_backend="pi",
            backend="pi",
            owned=True,
            start_ts=now - 200,
            cwd="/tmp/repo",
            log_path=None,
            sock_path=Path("/tmp/broker-old.sock"),
            last_chat_ts=now - 100,
        )
        distinct = Session(
            session_id="broker-other",
            thread_id="other-thread",
            broker_pid=102,
            codex_pid=202,
            agent_backend="pi",
            backend="pi",
            owned=True,
            start_ts=now - 40,
            cwd="/tmp/repo",
            log_path=None,
            sock_path=Path("/tmp/broker-other.sock"),
            last_chat_ts=now - 10,
        )
        mgr._sessions = {
            newest.session_id: newest,
            duplicate.session_id: duplicate,
            distinct.session_id: distinct,
        }

        rows = mgr.list_sessions()

        self.assertEqual(
            [row["session_id"] for row in rows],
            ["shared-thread", "other-thread"],
        )
        self.assertEqual(
            [row["thread_id"] for row in rows],
            ["shared-thread", "other-thread"],
        )

    def test_delete_session_kills_terminal_owned_and_clears_dependents(self) -> None:
        mgr = _make_manager()
        now = time.time()
        blocked = _session(sid="blocked", start_ts=now - 100, last_chat_ts=now - 10)
        target = _session(
            sid="target", start_ts=now - 200, last_chat_ts=now - 20, owned=False
        )
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
        s = _session(
            sid="target", start_ts=time.time() - 10, last_chat_ts=None, owned=False
        )
        mgr._sessions = {s.session_id: s}

        def _sock_call(*args, **kwargs):
            raise OSError("dead socket")

        mgr._sock_call = _sock_call  # type: ignore[method-assign]

        with patch.object(
            mgr, "_kill_session_via_pids", return_value=True
        ) as kill_via_pids:
            ok = mgr.kill_session("target")

        self.assertTrue(ok)
        kill_via_pids.assert_called_once_with(s)

    def test_kill_session_via_pids_prunes_stale_metadata_without_signals(self) -> None:
        mgr = _make_manager()
        s = _session(
            sid="target", start_ts=time.time() - 10, last_chat_ts=None, owned=False
        )

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

    def test_list_sessions_uses_start_ts_when_log_has_no_sidebar_relevant_message(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]
        with patch(
            "codoxear.server._last_conversation_ts_from_tail", return_value=None
        ):
            s = _session(sid="nologmsg", start_ts=123.0, last_chat_ts=None)
            s.log_path = Path("/tmp/fake-rollout.jsonl")
            mgr._sessions = {s.session_id: s}
            original_exists = Path.exists

            def _exists(path_obj):
                if str(path_obj) == "/tmp/fake-rollout.jsonl":
                    return True
                return original_exists(path_obj)

            with patch("pathlib.Path.exists", _exists):
                rows = mgr.list_sessions()

        self.assertEqual(rows[0]["updated_ts"], 123.0)

    def test_list_sessions_backfills_updated_ts_from_large_preexisting_log(
        self,
    ) -> None:
        mgr = _make_manager()
        mgr.idle_from_log = lambda _sid: True  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            log_path = (
                Path(td)
                / "rollout-2026-03-17T00-00-00-eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee.jsonl"
            )
            now = time.time()
            user_ts = now - 30
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "session_meta",
                                "payload": {"id": "current", "source": "cli"},
                            }
                        ),
                        json.dumps(
                            {
                                "type": "event_msg",
                                "payload": {
                                    "type": "user_message",
                                    "message": "real turn",
                                },
                                "ts": user_ts,
                            }
                        ),
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
                        json.dumps(
                            {
                                "type": "response_item",
                                "payload": {"type": "reasoning"},
                                "ts": now - 10,
                            }
                        ),
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
            log_path = (
                Path(td)
                / "rollout-2026-03-17T00-00-00-ffffffff-ffff-ffff-ffff-ffffffffffff.jsonl"
            )
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "session_meta",
                                "payload": {"id": "current", "source": "cli"},
                            }
                        ),
                        json.dumps(
                            {
                                "type": "event_msg",
                                "payload": {"type": "agent_reasoning"},
                                "ts": time.time(),
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            current = _session(sid="current", start_ts=123.0, last_chat_ts=None)
            current.log_path = log_path
            mgr._sessions = {current.session_id: current}

            with patch(
                "codoxear.server._last_conversation_ts_from_tail", return_value=None
            ) as backfill:
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

        self.assertEqual(
            mgr.recent_cwds(limit=4),
            [str(Path("/tmp/current").resolve(strict=False)), "/repo/ended"],
        )

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
            log_path = (
                Path(td)
                / "rollout-2026-03-17T00-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            )
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

    def test_load_cwd_groups_normalizes_loaded_keys(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            groups_path = Path(td) / "cwd_groups.json"
            cwd_raw = str(Path(td) / "project" / "foo" / ".." / "bar")
            expected_normalized = str(Path(cwd_raw).resolve(strict=False))
            groups_path.write_text(
                json.dumps({cwd_raw: {"label": " Demo ", "collapsed": True}}) + "\n",
                encoding="utf-8",
            )

            with patch("codoxear.server.CWD_GROUPS_PATH", groups_path):
                mgr._load_cwd_groups()

        self.assertEqual(
            mgr.cwd_groups_get(),
            {expected_normalized: {"label": "Demo", "collapsed": True}},
        )

    def test_load_cwd_groups_defaults_malformed_collapsed_values_to_false(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            groups_path = Path(td) / "cwd_groups.json"
            base = Path(td)
            truthy_string_cwd = str(base / "truthy-string")
            zero_string_cwd = str(base / "zero-string")
            integer_cwd = str(base / "integer")
            unlabeled_cwd = str(base / "unlabeled")
            groups_path.write_text(
                json.dumps(
                    {
                        truthy_string_cwd: {
                            "label": "Truthy String",
                            "collapsed": "false",
                        },
                        zero_string_cwd: {"label": "Zero String", "collapsed": "0"},
                        integer_cwd: {"label": "Integer", "collapsed": 1},
                        unlabeled_cwd: {"collapsed": "true"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("codoxear.server.CWD_GROUPS_PATH", groups_path):
                mgr._load_cwd_groups()

        self.assertEqual(
            mgr.cwd_groups_get(),
            {
                str(Path(truthy_string_cwd).resolve(strict=False)): {
                    "label": "Truthy String",
                    "collapsed": False,
                },
                str(Path(zero_string_cwd).resolve(strict=False)): {
                    "label": "Zero String",
                    "collapsed": False,
                },
                str(Path(integer_cwd).resolve(strict=False)): {
                    "label": "Integer",
                    "collapsed": False,
                },
            },
        )

    def test_load_cwd_groups_falls_back_to_empty_store_on_malformed_json(self) -> None:
        mgr = _make_manager()
        mgr._cwd_groups = {"/tmp/stale": {"label": "Stale", "collapsed": True}}
        with tempfile.TemporaryDirectory() as td:
            groups_path = Path(td) / "cwd_groups.json"
            groups_path.write_text("{not valid json\n", encoding="utf-8")

            with (
                patch("codoxear.server.CWD_GROUPS_PATH", groups_path),
                patch("codoxear.server.LOG.warning") as warning,
            ):
                mgr._load_cwd_groups()

        self.assertEqual(mgr.cwd_groups_get(), {})
        warning.assert_called_once()

    def test_cwd_groups_get_returns_entry_copies(self) -> None:
        mgr = _make_manager()
        mgr._cwd_groups = {"/tmp/project": {"label": "Project", "collapsed": True}}

        groups = mgr.cwd_groups_get()
        groups["/tmp/project"]["label"] = "Mutated"
        groups["/tmp/project"]["collapsed"] = False

        self.assertEqual(
            mgr._cwd_groups,
            {"/tmp/project": {"label": "Project", "collapsed": True}},
        )

    def test_cwd_group_set_normalizes_path_and_round_trips_metadata(self) -> None:
        mgr = _make_manager()
        mgr._recent_cwds = {"/tmp/project/bar": time.time()}
        cwd_raw = "/tmp/project/foo/../bar"
        expected_normalized = str(Path(cwd_raw).resolve(strict=False))
        normalized, meta = mgr.cwd_group_set(
            cwd=cwd_raw, label=" My Project ", collapsed=True
        )

        self.assertEqual(normalized, expected_normalized)
        self.assertEqual(meta, {"label": "My Project", "collapsed": True})

        # Verify retrieval
        all_groups = mgr.cwd_groups_get()
        self.assertEqual(all_groups[normalized], meta)

    def test_cwd_group_set_canonicalizes_trimmed_path(self) -> None:
        mgr = _make_manager()
        cwd_raw = " /tmp/project/foo/../bar "
        expected_normalized = str(Path(cwd_raw.strip()).resolve(strict=False))
        mgr._recent_cwds = {expected_normalized: time.time()}

        normalized, meta = mgr.cwd_group_set(
            cwd=cwd_raw, label="Trimmed", collapsed=True
        )

        self.assertEqual(normalized, expected_normalized)
        self.assertEqual(meta, {"label": "Trimmed", "collapsed": True})

    def test_list_sessions_canonicalizes_session_cwds_for_group_metadata(self) -> None:
        mgr = _make_manager()
        with tempfile.TemporaryDirectory() as td:
            raw_cwd = str(Path(td) / "project" / "." / "docs" / ".." / "docs")
            expected_cwd = str(Path(raw_cwd).resolve(strict=False))
            session = _session(sid="docs", start_ts=time.time())
            session.cwd = f"  {raw_cwd}  "
            mgr._sessions = {session.session_id: session}

            with patch("codoxear.server._current_git_branch", return_value=None):
                rows = mgr.list_sessions()

        self.assertEqual(rows[0]["cwd"], expected_cwd)
        self.assertIn(expected_cwd, mgr._recent_cwds)

    def test_cwd_group_set_returns_meta_copy(self) -> None:
        mgr = _make_manager()
        mgr._recent_cwds = {"/tmp/project": time.time()}
        normalized, meta = mgr.cwd_group_set(
            cwd="/tmp/project", label="Project", collapsed=True
        )

        meta["label"] = "Mutated"
        meta["collapsed"] = False

        self.assertEqual(
            mgr._cwd_groups[normalized],
            {"label": "Project", "collapsed": True},
        )

    def test_cwd_group_set_clears_label_preserves_existing_collapsed(self) -> None:
        mgr = _make_manager()
        cwd = "/tmp/foo"
        mgr._recent_cwds = {cwd: time.time()}
        normalized, _ = mgr.cwd_group_set(cwd=cwd, label="Foo", collapsed=True)

        normalized_again, meta = mgr.cwd_group_set(cwd=cwd, label="")

        self.assertEqual(normalized_again, normalized)
        self.assertEqual(meta, {"label": "", "collapsed": True})
        self.assertEqual(mgr.cwd_groups_get()[normalized], meta)

    def test_cwd_group_set_updates_collapsed_preserves_existing_label(self) -> None:
        mgr = _make_manager()
        cwd = "/tmp/foo"
        mgr._recent_cwds = {cwd: time.time()}
        normalized, _ = mgr.cwd_group_set(cwd=cwd, label="Foo")

        normalized_again, meta = mgr.cwd_group_set(cwd=cwd, collapsed=True)

        self.assertEqual(normalized_again, normalized)
        self.assertEqual(meta, {"label": "Foo", "collapsed": True})
        self.assertEqual(mgr.cwd_groups_get()[normalized], meta)

    def test_cwd_group_set_drops_empty_default_entries_from_serialized_store(
        self,
    ) -> None:
        mgr = _make_manager()
        cwd = "/tmp/foo"
        with tempfile.TemporaryDirectory() as td:
            groups_path = Path(td) / "cwd_groups.json"
            mgr._recent_cwds = {cwd: time.time()}
            mgr._save_cwd_groups = SessionManager._save_cwd_groups.__get__(
                mgr, SessionManager
            )

            with (
                patch("codoxear.server.APP_DIR", Path(td)),
                patch("codoxear.server.CWD_GROUPS_PATH", groups_path),
            ):
                normalized, _ = mgr.cwd_group_set(cwd=cwd, label="Foo", collapsed=True)
                self.assertEqual(
                    json.loads(groups_path.read_text(encoding="utf-8")),
                    {normalized: {"label": "Foo", "collapsed": True}},
                )

                mgr.cwd_group_set(cwd=cwd, label="", collapsed=False)

                self.assertEqual(
                    json.loads(groups_path.read_text(encoding="utf-8")), {}
                )

        self.assertNotIn(normalized, mgr.cwd_groups_get())

    def test_cwd_group_set_rejects_non_boolean_collapsed(self) -> None:
        mgr = _make_manager()
        mgr._recent_cwds = {"/tmp": time.time()}
        with self.assertRaisesRegex(ValueError, "collapsed must be a boolean"):
            mgr.cwd_group_set(cwd="/tmp", collapsed="not-a-bool")  # type: ignore

    def test_cwd_group_set_rejects_non_string_label(self) -> None:
        mgr = _make_manager()
        mgr._recent_cwds = {"/tmp": time.time()}

        with self.assertRaisesRegex(ValueError, "label must be a string"):
            mgr.cwd_group_set(cwd="/tmp", label=123)  # type: ignore[arg-type]

    def test_cwd_group_set_rejects_unknown_cwd(self) -> None:
        mgr = _make_manager()
        mgr._recent_cwds = {"/tmp/known": time.time()}

        with self.assertRaisesRegex(
            ValueError, "cwd is not a known session working directory"
        ):
            mgr.cwd_group_set(cwd="/tmp/unknown", label="Unknown")

    def test_cwd_group_set_accepts_known_live_session_cwd(self) -> None:
        mgr = _make_manager()
        current = _session(
            sid="current", start_ts=time.time() - 10, last_chat_ts=time.time() - 5
        )
        mgr._sessions = {current.session_id: current}
        expected_normalized = str(Path(current.cwd).resolve(strict=False))

        normalized, meta = mgr.cwd_group_set(
            cwd=current.cwd, label="Current", collapsed=True
        )

        self.assertEqual(normalized, expected_normalized)
        self.assertEqual(meta, {"label": "Current", "collapsed": True})

    def test_load_recent_cwds_skips_stale_directories(self) -> None:
        mgr = _make_manager()
        mgr._prune_missing_workspace_dirs = True
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid = root / "valid"
            valid.mkdir()
            stale = root / "stale"
            recent_path = root / "recent_cwds.json"
            recent_path.write_text(
                json.dumps({str(valid): 20, str(stale): 10}), encoding="utf-8"
            )

            with patch("codoxear.server.RECENT_CWD_PATH", recent_path):
                mgr._load_recent_cwds()
                self.assertEqual(mgr.recent_cwds(), [str(valid)])

    def test_load_cwd_groups_skips_stale_directories(self) -> None:
        mgr = _make_manager()
        mgr._prune_missing_workspace_dirs = True
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid = root / "valid"
            valid.mkdir()
            stale = root / "stale"
            groups_path = root / "cwd_groups.json"
            groups_path.write_text(
                json.dumps(
                    {
                        str(valid): {"label": "Valid", "collapsed": False},
                        str(stale): {"label": "Stale", "collapsed": True},
                    }
                ),
                encoding="utf-8",
            )

            with patch("codoxear.server.CWD_GROUPS_PATH", groups_path):
                mgr._load_cwd_groups()
                self.assertEqual(
                    mgr.cwd_groups_get(),
                    {str(valid.resolve()): {"label": "Valid", "collapsed": False}},
                )

    def test_cwd_group_set_allows_clearing_stale_group(self) -> None:
        mgr = _make_manager()
        mgr._prune_missing_workspace_dirs = True
        with tempfile.TemporaryDirectory() as td:
            stale = Path(td) / "deleted-project"
            normalized = str(stale.resolve())
            mgr._cwd_groups = {normalized: {"label": "Old", "collapsed": True}}

            returned_cwd, meta = mgr.cwd_group_set(
                cwd=normalized, label="", collapsed=False
            )

        self.assertEqual(returned_cwd, normalized)
        self.assertEqual(meta, {"label": "", "collapsed": False})
        self.assertNotIn(normalized, mgr.cwd_groups_get())


if __name__ == "__main__":
    unittest.main()
