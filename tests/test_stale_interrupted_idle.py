"""R1 fix: stale ``interrupted_idle`` must not make the session listing project
idle for a same-log turn that resumed after an interrupt.

The listing path (``/api/sessions``) reads the stored ``interrupted_idle``
boolean captured from the last broker poll. If a new turn starts on the same
log (terminal / another client) while the session is deselected, the broker has
already cleared its interrupt state but the stored boolean is stale, so the
interrupted-idle override forced ``busy=False`` against a non-idle log.

The fix records the log byte offset at the moment ``interrupted_idle`` becomes
True (``interrupted_idle_log_off``) and the log watcher
(``SessionLogRuntimeCoordinator.update_meta_counters``) clears the stale boolean
once it observes post-interrupt user/assistant activity. These tests exercise
the real coordinator and real ``_analyze_log_chunk``.
"""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.rollout_idle import _analyze_log_chunk
from codoxear.rollout_jsonl import _read_jsonl_records_from_offset
from codoxear.session_list import SessionListCoordinator
from codoxear.session_log_runtime import SessionLogRuntimeCoordinator
from codoxear.session_model import Session
from codoxear.session_runtime import ListingRuntimeProbes
from codoxear.session_runtime import log_path_size_or_none
from codoxear.session_runtime import set_session_interrupted_idle
from codoxear.session_store import SessionStore
from codoxear.session_store import SessionStorePaths


def _append_jsonl(path: Path, objs: list[dict]) -> int:
    """Append objs and return the new file size."""
    with path.open("a", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj) + "\n")
    return int(path.stat().st_size)


def _read_jsonl_objs(path: Path, offset: int, max_bytes: int = 256 * 1024) -> tuple[list[dict], int]:
    records, new_off = _read_jsonl_records_from_offset(path, offset, max_bytes=max_bytes)
    return [r.obj for r in records], new_off


def _make_session(log_path: Path, *, interrupted_idle: bool = False) -> Session:
    s = Session(
        session_id="broker-1",
        thread_id="broker-1",
        broker_pid=1,
        codex_pid=2,
        agent_backend="codex",
        owned=False,
        start_ts=100.0,
        cwd="/tmp",
        log_path=log_path,
        sock_path=Path("/tmp/broker-1.sock"),
        busy=False,
        queue_len=0,
    )
    # Seed meta_log_off at the current log size (matches discovery / refresh,
    # which call reset_log_caches with the live log size).
    s.meta_log_off = int(log_path.stat().st_size)
    if interrupted_idle:
        # Simulate get_state confirming an interrupt right now: the helper
        # captures the current log size as the post-interrupt baseline.
        set_session_interrupted_idle(s, True)
    return s


def _log_runtime(sessions: dict[str, Session]) -> SessionLogRuntimeCoordinator:
    lock = threading.Lock()
    return SessionLogRuntimeCoordinator(
        lock=lock,
        sessions=lambda: sessions,
        analyze_log_chunk=_analyze_log_chunk,
        turn_context_run_settings=lambda _payload: (None, None),
        compute_idle_from_log=lambda _path: True,
        read_jsonl_from_offset=_read_jsonl_objs,
        find_latest_token_update=lambda _path: None,
    )


class TestSetSessionInterruptedIdle(unittest.TestCase):
    def test_records_log_size_baseline_when_set_true(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"s"}}\n', encoding="utf-8")
            size = int(log_path.stat().st_size)
            s = Session(
                session_id="s", thread_id="s", broker_pid=1, codex_pid=2,
                agent_backend="codex", owned=False, start_ts=1.0, cwd="/tmp",
                log_path=log_path, sock_path=Path("/tmp/s.sock"),
            )
            set_session_interrupted_idle(s, True)
            self.assertTrue(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, size)

    def test_resets_baseline_when_cleared(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"x":1}\n', encoding="utf-8")
            s = Session(
                session_id="s", thread_id="s", broker_pid=1, codex_pid=2,
                agent_backend="codex", owned=False, start_ts=1.0, cwd="/tmp",
                log_path=log_path, sock_path=Path("/tmp/s.sock"),
            )
            set_session_interrupted_idle(s, True)
            self.assertGreater(s.interrupted_idle_log_off, 0)
            set_session_interrupted_idle(s, False)
            self.assertFalse(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, 0)

    def test_missing_log_path_records_zero_baseline(self) -> None:
        s = Session(
            session_id="s", thread_id="s", broker_pid=1, codex_pid=2,
            agent_backend="codex", owned=False, start_ts=1.0, cwd="/tmp",
            log_path=None, sock_path=Path("/tmp/s.sock"),
        )
        set_session_interrupted_idle(s, True)
        self.assertTrue(s.interrupted_idle)
        self.assertEqual(s.interrupted_idle_log_off, 0)


class TestStaleInterruptedIdleInvalidation(unittest.TestCase):
    def _interrupted_turn_log(self, log_path: Path) -> None:
        """A turn whose last row is a non-final assistant fragment (interrupted)."""
        log_path.write_text(
            "".join(
                json.dumps(o) + "\n"
                for o in [
                    {"type": "session_meta", "payload": {"id": "broker-1", "source": "cli"}},
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "first"}, "ts": 10.0},
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "working"}],
                        },
                        "ts": 11.0,
                    },
                ]
            ),
            encoding="utf-8",
        )

    def test_post_interrupt_user_message_clears_stale_interrupted_idle(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            self._interrupted_turn_log(log_path)
            s = _make_session(log_path, interrupted_idle=True)
            runtime = _log_runtime({"broker-1": s})

            # Simulate a new turn started from terminal while deselected.
            _append_jsonl(
                log_path,
                [{"type": "event_msg", "payload": {"type": "user_message", "message": "second"}, "ts": 20.0}],
            )

            runtime.update_meta_counters()

            self.assertFalse(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, 0)

    def test_post_interrupt_assistant_activity_clears_stale_interrupted_idle(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            self._interrupted_turn_log(log_path)
            s = _make_session(log_path, interrupted_idle=True)
            runtime = _log_runtime({"broker-1": s})

            # Agent resumes producing (reasoning) after the interrupt.
            _append_jsonl(
                log_path,
                [{"type": "event_msg", "payload": {"type": "agent_reasoning"}, "ts": 20.0}],
            )

            runtime.update_meta_counters()

            self.assertFalse(s.interrupted_idle)

    def test_immediate_interrupt_nonfinal_tail_preserves_override(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            self._interrupted_turn_log(log_path)
            size_at_interrupt = int(log_path.stat().st_size)
            s = _make_session(log_path, interrupted_idle=True)
            # meta_log_off is at the interrupt point; nothing new has arrived.
            self.assertEqual(s.meta_log_off, size_at_interrupt)
            runtime = _log_runtime({"broker-1": s})

            runtime.update_meta_counters()

            # No post-interrupt activity -> override input stays alive.
            self.assertTrue(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, size_at_interrupt)

    def test_log_shrink_does_not_clear_interrupted_idle(self) -> None:
        # Defensive: if the log is truncated below the recorded baseline, the
        # watcher must not clear (conservative; state reset is handled elsewhere).
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            self._interrupted_turn_log(log_path)
            s = _make_session(log_path, interrupted_idle=True)
            baseline = s.interrupted_idle_log_off
            runtime = _log_runtime({"broker-1": s})

            # Truncate the log to a single row below the baseline.
            log_path.write_text('{"type":"session_meta","payload":{"id":"broker-1"}}\n', encoding="utf-8")
            self.assertLess(int(log_path.stat().st_size), baseline)

            runtime.update_meta_counters()

            self.assertTrue(s.interrupted_idle)


def _store(tmp_path: Path) -> SessionStore:
    return SessionStore(
        paths=SessionStorePaths(
            aliases=tmp_path / "aliases.json",
            sidebar_meta=tmp_path / "sidebar.json",
            hidden_sessions=tmp_path / "hidden.json",
            files=tmp_path / "files.json",
            queues=tmp_path / "queues.json",
            pending_attachments=tmp_path / "pending.json",
            commit_unknown_sends=tmp_path / "unknown.json",
            recent_cwds=tmp_path / "recent.json",
            unattended=tmp_path / "unattended.json",
        ),
        file_history_max=5,
        recent_cwd_max=5,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_alias=lambda value: value if isinstance(value, str) else "",
        clean_priority_offset=lambda value: float(value or 0.0),
        clean_snooze_until=lambda value: float(value) if value not in (None, "", 0) else None,
        clean_dependency_session_id=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        clean_recent_cwd=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        clean_commit_unknown_send_record=lambda value: value if isinstance(value, dict) else None,
    )


def _list_coordinator(
    *,
    sessions: dict[str, Session],
    tmp_path: Path,
    log_runtime: SessionLogRuntimeCoordinator,
    idle_from_log_path,
) -> SessionListCoordinator:
    probes = ListingRuntimeProbes(
        last_conversation_ts_from_tail=lambda _path: None,
        read_run_settings_from_log=lambda _path, _backend: (None, None, None),
        log_size_or_none=log_path_size_or_none,
        send_boundary_unresolved=lambda _sid, _lp, _ls: False,
        idle_from_log_path=idle_from_log_path,
        current_git_branch=lambda _path: None,
    )
    return SessionListCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        queues=lambda: {},
        unattended=lambda: {},
        aliases=lambda: {},
        hidden_sessions=lambda: set(),
        commit_unknown_sends=lambda: {},
        store=_store(tmp_path),
        discover_existing_if_stale=lambda: None,
        prune_dead_sessions=lambda: None,
        update_meta_counters=log_runtime.update_meta_counters,
        save_files=lambda: None,
        save_sidebar_meta=lambda: None,
        save_recent_cwds=lambda: None,
        now=lambda: 200.0,
        runtime_probes=probes,
        include_launch_attempts=lambda: False,
        read_launch_attempts=lambda: [],
        launch_attempt_row=lambda _record: None,
        clean_unattended_cooldown_minutes=lambda value: int(value),
        clean_unattended_remaining_injections=lambda value, *, allow_zero=False: int(value),
        provider_choice_for_settings=lambda **_kwargs: "openai-api",
        resolve_session_cwd=lambda cwd: Path(cwd),
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        priority_half_life_seconds=100.0,
        priority_bucket_seconds=10.0,
    )


class TestListingProjectionAfterInvalidation(unittest.TestCase):
    def test_listing_projects_busy_after_new_turn_clears_stale_override(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text(
                "".join(
                    json.dumps(o) + "\n"
                    for o in [
                        {"type": "session_meta", "payload": {"id": "broker-1", "source": "cli"}},
                        {"type": "event_msg", "payload": {"type": "user_message", "message": "first"}, "ts": 10.0},
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "working"}],
                            },
                            "ts": 11.0,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            s = _make_session(log_path, interrupted_idle=True)
            sessions = {"broker-1": s}
            log_runtime = _log_runtime(sessions)

            # New turn resumes on the same log while deselected.
            _append_jsonl(
                log_path,
                [{"type": "event_msg", "payload": {"type": "user_message", "message": "second"}, "ts": 20.0}],
            )

            coordinator = _list_coordinator(
                sessions=sessions,
                tmp_path=Path(td),
                log_runtime=log_runtime,
                idle_from_log_path=lambda _sid, _path: False,
            )
            out = coordinator.list_sessions()

            self.assertEqual(len(out), 1)
            # Non-idle log + stale override invalidated -> busy must be True.
            self.assertIs(out[0]["busy"], True)
            self.assertFalse(s.interrupted_idle)

    def test_listing_keeps_idle_projection_for_immediate_interrupt(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text(
                "".join(
                    json.dumps(o) + "\n"
                    for o in [
                        {"type": "session_meta", "payload": {"id": "broker-1", "source": "cli"}},
                        {"type": "event_msg", "payload": {"type": "user_message", "message": "first"}, "ts": 10.0},
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "working"}],
                            },
                            "ts": 11.0,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            s = _make_session(log_path, interrupted_idle=True)
            sessions = {"broker-1": s}
            log_runtime = _log_runtime(sessions)

            coordinator = _list_coordinator(
                sessions=sessions,
                tmp_path=Path(td),
                log_runtime=log_runtime,
                # Log is non-idle only because the interrupted turn left it
                # non-final; the override must still apply.
                idle_from_log_path=lambda _sid, _path: False,
            )
            out = coordinator.list_sessions()

            self.assertEqual(len(out), 1)
            # Immediate interrupt: override preserved -> busy False.
            self.assertIs(out[0]["busy"], False)
            self.assertTrue(s.interrupted_idle)


if __name__ == "__main__":
    unittest.main()
