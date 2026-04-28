import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear import server
from codoxear.server import Session
from codoxear.server import SessionManager
from codoxear.util import append_launch_attempt
from codoxear.util import read_launch_attempts


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
    mgr._include_launch_attempts = True
    mgr._discover_existing_if_stale = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._prune_dead_sessions = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._update_meta_counters = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_hidden_sessions = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_aliases = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_sidebar_meta = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_harness = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_files = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mgr._save_queues = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return mgr


class TestLaunchProvenance(unittest.TestCase):
    def test_launch_attempt_log_collapses_to_latest_state(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "session_launches.jsonl"
            created = append_launch_attempt(
                {
                    "launch_id": "launch-a",
                    "state": "starting",
                    "agent_backend": "codex",
                    "cwd": "/tmp/work",
                    "spawn_nonce": "nonce-a",
                    "created_ts": time.time(),
                },
                path=path,
            )
            append_launch_attempt(
                {
                    "launch_id": created["launch_id"],
                    "state": "failed",
                    "stage": "broker_metadata",
                    "error": "tmux launch did not publish broker metadata within 3.0s",
                    "agent_backend": "codex",
                    "cwd": "/tmp/work",
                    "spawn_nonce": "nonce-a",
                    "created_ts": created["created_ts"],
                    "updated_ts": time.time() + 1.0,
                },
                path=path,
            )

            rows = read_launch_attempts(path=path, max_records=10, max_age_s=3600)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["launch_id"], "launch-a")
        self.assertEqual(rows[0]["state"], "failed")
        self.assertEqual(rows[0]["stage"], "broker_metadata")
        self.assertEqual(rows[0]["spawn_nonce"], "nonce-a")

    def test_list_sessions_exposes_recent_failed_launch_as_session_row(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td, patch.object(server, "LAUNCH_ATTEMPTS_PATH", Path(td) / "launches.jsonl"):
            append_launch_attempt(
                {
                    "launch_id": "launch-pi",
                    "state": "failed",
                    "stage": "pty_fork",
                    "error": "pty fork failed before agent start: OSError: out of pty devices",
                    "agent_backend": "pi",
                    "cwd": "/tmp/pi-work",
                    "transport": "tmux",
                    "tmux_session": "codoxear",
                    "tmux_window": "pi-work-abc123",
                    "model_provider": "macaron",
                    "model": "gpt-5.4",
                    "reasoning_effort": "medium",
                    "created_ts": time.time(),
                },
                path=server.LAUNCH_ATTEMPTS_PATH,
            )

            rows = mgr.list_sessions()

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["launch_state"], "failed")
        self.assertEqual(row["launch_stage"], "pty_fork")
        self.assertIn("out of pty devices", row["launch_error"])
        self.assertEqual(row["launch_id"], "launch-pi")
        self.assertEqual(row["agent_backend"], "pi")
        self.assertEqual(row["provider_choice"], "macaron")
        self.assertEqual(row["busy"], False)
        self.assertEqual(row["final_priority"], 1.0)

    def test_list_sessions_exposes_pending_launch_as_session_row(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td, patch.object(server, "LAUNCH_ATTEMPTS_PATH", Path(td) / "launches.jsonl"):
            append_launch_attempt(
                {
                    "launch_id": "launch-pending",
                    "state": "tmux_pane_created",
                    "agent_backend": "codex",
                    "cwd": "/tmp/work",
                    "transport": "tmux",
                    "tmux_session": "codoxear",
                    "tmux_window": "work-123abc",
                    "created_ts": time.time(),
                },
                path=server.LAUNCH_ATTEMPTS_PATH,
            )

            rows = mgr.list_sessions()

        self.assertEqual(rows[0]["session_id"], "launch-pending")
        self.assertEqual(rows[0]["launch_state"], "tmux_pane_created")
        self.assertEqual(rows[0]["busy"], False)

    def test_delete_real_session_hides_stale_pending_launch_row(self) -> None:
        mgr = _make_manager()
        now = time.time()
        with TemporaryDirectory() as td, patch.object(server, "LAUNCH_ATTEMPTS_PATH", Path(td) / "launches.jsonl"):
            append_launch_attempt(
                {
                    "launch_id": "launch-live",
                    "state": "starting",
                    "agent_backend": "codex",
                    "cwd": "/tmp/work",
                    "transport": "tmux",
                    "spawn_nonce": "nonce-live",
                    "created_ts": now,
                    "updated_ts": now,
                },
                path=server.LAUNCH_ATTEMPTS_PATH,
            )
            mgr._sessions = {
                "broker-live": Session(
                    session_id="broker-live",
                    thread_id="broker-live",
                    broker_pid=1234,
                    codex_pid=1235,
                    agent_backend="codex",
                    owned=True,
                    start_ts=now,
                    cwd="/tmp/work",
                    log_path=None,
                    sock_path=Path(td) / "broker-live.sock",
                    transport="tmux",
                    launch_id="launch-live",
                    spawn_nonce="nonce-live",
                )
            }
            mgr._sock_call = lambda *args, **kwargs: {"ok": True}  # type: ignore[method-assign]

            before = mgr.list_sessions()
            self.assertEqual([row["session_id"] for row in before], ["broker-live"])

            self.assertTrue(mgr.delete_session("broker-live"))
            after = mgr.list_sessions()

        self.assertEqual(after, [])
        self.assertIn("launch-live", mgr._hidden_sessions)

    def test_prune_preserves_specific_existing_launch_failure(self) -> None:
        mgr = _make_manager()
        now = time.time()
        with TemporaryDirectory() as td, patch.object(server, "LAUNCH_ATTEMPTS_PATH", Path(td) / "launches.jsonl"):
            append_launch_attempt(
                {
                    "launch_id": "launch-shell",
                    "state": "failed",
                    "stage": "shell_startup",
                    "error": "shell startup blocked before agent exec",
                    "agent_backend": "codex",
                    "cwd": "/tmp/work",
                    "transport": "tmux",
                    "created_ts": now,
                    "updated_ts": now,
                },
                path=server.LAUNCH_ATTEMPTS_PATH,
            )
            mgr._sessions = {
                "broker-dead": Session(
                    session_id="broker-dead",
                    thread_id="broker-dead",
                    broker_pid=1234,
                    codex_pid=1235,
                    agent_backend="codex",
                    owned=True,
                    start_ts=now,
                    cwd="/tmp/work",
                    log_path=None,
                    sock_path=Path(td) / "missing.sock",
                    transport="tmux",
                    launch_id="launch-shell",
                )
            }

            SessionManager._prune_dead_sessions(mgr)
            rows = read_launch_attempts(path=server.LAUNCH_ATTEMPTS_PATH, max_records=10, max_age_s=3600)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], "shell_startup")

    def test_list_sessions_omits_successful_launch_attempt_without_active_session(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td, patch.object(server, "LAUNCH_ATTEMPTS_PATH", Path(td) / "launches.jsonl"):
            append_launch_attempt(
                {
                    "launch_id": "launch-bound",
                    "state": "broker_meta_bound",
                    "agent_backend": "codex",
                    "cwd": "/tmp/work",
                    "transport": "tmux",
                    "tmux_session": "codoxear",
                    "tmux_window": "work-123abc",
                    "broker_pid": 1234,
                    "created_ts": time.time(),
                },
                path=server.LAUNCH_ATTEMPTS_PATH,
            )
            append_launch_attempt(
                {
                    "launch_id": "launch-spawned",
                    "state": "broker_spawned",
                    "agent_backend": "codex",
                    "cwd": "/tmp/direct",
                    "transport": "direct",
                    "broker_pid": 1235,
                    "created_ts": time.time(),
                },
                path=server.LAUNCH_ATTEMPTS_PATH,
            )

            rows = mgr.list_sessions()

        self.assertEqual(rows, [])

    def test_delete_session_dismisses_launch_attempt_row(self) -> None:
        mgr = _make_manager()
        with TemporaryDirectory() as td, patch.object(server, "LAUNCH_ATTEMPTS_PATH", Path(td) / "launches.jsonl"):
            rec = append_launch_attempt(
                {
                    "launch_id": "launch-dead",
                    "state": "failed",
                    "stage": "broker_early_exit",
                    "error": "broker exited early",
                    "agent_backend": "codex",
                    "cwd": "/tmp/work",
                    "created_ts": time.time(),
                },
                path=server.LAUNCH_ATTEMPTS_PATH,
            )
            launch_id = str(rec["launch_id"])

            self.assertTrue(mgr.delete_session(launch_id))
            rows = mgr.list_sessions()

        self.assertEqual(rows, [])
        self.assertIn(launch_id, mgr._hidden_sessions)


if __name__ == "__main__":
    unittest.main()
