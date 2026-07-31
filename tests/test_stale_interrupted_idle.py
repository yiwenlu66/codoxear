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
from codoxear.session_discovery import DiscoveryRegistration
from codoxear.session_discovery_registry import SessionDiscoveryRegistryCoordinator
from codoxear.session_list import SessionListCoordinator
from codoxear.session_log_runtime import SessionLogRuntimeCoordinator
from codoxear.session_model import Session
from codoxear.session_runtime import ListingRuntimeProbes
from codoxear.session_runtime import log_path_size_or_none
from codoxear.session_runtime import reset_session_log_caches
from codoxear.session_runtime import set_session_interrupted_idle
from codoxear.session_runtime import suppress_session_interrupted_idle
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

    def test_repeated_true_preserves_existing_baseline(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"x":1}\n', encoding="utf-8")
            s = Session(
                session_id="s", thread_id="s", broker_pid=1, codex_pid=2,
                agent_backend="codex", owned=False, start_ts=1.0, cwd="/tmp",
                log_path=log_path, sock_path=Path("/tmp/s.sock"),
            )
            set_session_interrupted_idle(s, True)
            baseline = s.interrupted_idle_log_off
            with log_path.open("a", encoding="utf-8") as f:
                f.write('{"x":2}\n')
            set_session_interrupted_idle(s, True)
            self.assertEqual(s.interrupted_idle_log_off, baseline)

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
            first_baseline = s.interrupted_idle_log_off
            self.assertGreater(first_baseline, 0)
            set_session_interrupted_idle(s, False)
            self.assertFalse(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, 0)
            with log_path.open("a", encoding="utf-8") as f:
                f.write('{"x":2}\n')
            set_session_interrupted_idle(s, True)
            self.assertGreater(s.interrupted_idle_log_off, first_baseline)

    def test_suppressed_stale_true_stays_cleared_until_broker_false(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"x":1}\n', encoding="utf-8")
            s = Session(
                session_id="s", thread_id="s", broker_pid=1, codex_pid=2,
                agent_backend="codex", owned=False, start_ts=1.0, cwd="/tmp",
                log_path=log_path, sock_path=Path("/tmp/s.sock"),
            )
            set_session_interrupted_idle(s, True)
            suppress_session_interrupted_idle(s)
            set_session_interrupted_idle(s, True)
            self.assertFalse(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, 0)
            self.assertTrue(s.interrupted_idle_suppressed)
            set_session_interrupted_idle(s, False)
            set_session_interrupted_idle(s, True)
            self.assertTrue(s.interrupted_idle)
            self.assertGreater(s.interrupted_idle_log_off, 0)
            self.assertFalse(s.interrupted_idle_suppressed)

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
    prune_dead_sessions=None,
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
        prune_dead_sessions=prune_dead_sessions or (lambda: None),
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

    def test_real_listing_order_clears_stale_override_after_prune_refresh(self) -> None:
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

            _append_jsonl(
                log_path,
                [{"type": "event_msg", "payload": {"type": "user_message", "message": "second"}, "ts": 20.0}],
            )

            def stale_prune_refresh() -> None:
                # Real listing order refreshes broker state before log counters.
                # The stale socket still reports interrupted_idle=True. The
                # baseline must not move forward past the newly appended row.
                set_session_interrupted_idle(s, True)

            coordinator = _list_coordinator(
                sessions=sessions,
                tmp_path=Path(td),
                log_runtime=log_runtime,
                idle_from_log_path=lambda _sid, _path: False,
                prune_dead_sessions=stale_prune_refresh,
            )
            out = coordinator.list_sessions()

            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], True)
            self.assertFalse(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, 0)
            self.assertTrue(s.interrupted_idle_suppressed)

            out_again = coordinator.list_sessions()
            self.assertEqual(len(out_again), 1)
            self.assertIs(out_again[0]["busy"], True)
            self.assertFalse(s.interrupted_idle)
            self.assertTrue(s.interrupted_idle_suppressed)

    def test_real_listing_order_keeps_immediate_interrupt_idle_after_prune_refresh(self) -> None:
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
            baseline = s.interrupted_idle_log_off

            coordinator = _list_coordinator(
                sessions=sessions,
                tmp_path=Path(td),
                log_runtime=log_runtime,
                idle_from_log_path=lambda _sid, _path: False,
                prune_dead_sessions=lambda: set_session_interrupted_idle(s, True),
            )
            out = coordinator.list_sessions()

            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], False)
            self.assertTrue(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, baseline)

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


def _discovery_registration(
    *,
    session: Session,
    log_path: Path,
    interrupted_idle: bool,
    meta_log_off: int,
) -> DiscoveryRegistration:
    """Build a registration that mirrors what discovery captures from a socket."""
    return DiscoveryRegistration(
        session_id=session.session_id,
        thread_id=session.thread_id,
        broker_pid=session.broker_pid,
        codex_pid=session.codex_pid,
        agent_backend=session.agent_backend,
        owned=session.owned,
        transport=session.transport,
        start_ts=session.start_ts,
        cwd=session.cwd,
        log_path=log_path,
        sock_path=session.sock_path,
        busy=session.busy,
        queue_len=session.queue_len,
        token=session.token,
        meta_log_off=meta_log_off,
        model_provider=session.model_provider,
        preferred_auth_method=session.preferred_auth_method,
        model=session.model,
        reasoning_effort=session.reasoning_effort,
        service_tier=session.service_tier,
        tmux_session=session.tmux_session,
        tmux_window=session.tmux_window,
        launch_id=session.launch_id,
        spawn_nonce=session.spawn_nonce,
        resume_session_id=session.resume_session_id,
        sync_send_supported=session.sync_send_supported,
        key_write_errors_supported=session.key_write_errors_supported,
        interrupted_idle=interrupted_idle,
    )


def _discovery_registry(
    *, sessions: dict[str, Session], lock: threading.Lock
) -> SessionDiscoveryRegistryCoordinator:
    return SessionDiscoveryRegistryCoordinator(
        lock=lock,
        sessions=lambda: sessions,
        pending_attachment_ids=lambda: set(),
        commit_unknown_sends=lambda: {},
        reset_log_caches=lambda session, log_off: reset_session_log_caches(session, meta_log_off=log_off),
        record_launch_attempt=lambda _record: None,
        prune_stale_socket_without_metadata=lambda _sid, _sock: None,
        unhide_session=lambda _sid: None,
        unlink_quiet=lambda _path: None,
        remember_recent_cwd=lambda *_a, **_k: False,
        save_recent_cwds=lambda: None,
    )


class TestDiscoveryRefreshPreservesInterruptBaseline(unittest.TestCase):
    """Discovery refresh must not re-baseline a stale interrupted-idle override
    past post-interrupt resumed activity. This is the discovery-first timing the
    broker/prune fix (R1) did not cover: discovery runs before
    ``update_meta_counters`` and the broker still reports the old
    ``interrupted_idle=True``.
    """

    def _interrupted_turn_log(self, log_path: Path) -> None:
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

    def test_discovery_refresh_before_counters_does_not_rebaseline_past_resumed_activity(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            self._interrupted_turn_log(log_path)
            s = _make_session(log_path, interrupted_idle=True)
            baseline_at_interrupt = s.interrupted_idle_log_off
            self.assertGreater(baseline_at_interrupt, 0)
            sessions = {"broker-1": s}
            lock = threading.Lock()
            log_runtime = _log_runtime(sessions)
            registry = _discovery_registry(sessions=sessions, lock=lock)

            # Resumed turn starts on the same log while deselected.
            _append_jsonl(
                log_path,
                [{"type": "event_msg", "payload": {"type": "user_message", "message": "second"}, "ts": 20.0}],
            )
            size_after_resume = int(log_path.stat().st_size)
            self.assertGreater(size_after_resume, baseline_at_interrupt)

            # Discovery runs FIRST: broker still reports the stale
            # interrupted_idle=True, and meta_log_off is the current log size
            # (past the resumed user_message). upsert_registration must not move
            # the baseline forward.
            reg = _discovery_registration(
                session=s,
                log_path=log_path,
                interrupted_idle=True,
                meta_log_off=size_after_resume,
            )
            registry.upsert_registration(reg)
            self.assertTrue(s.interrupted_idle)
            # Baseline preserved at the interrupt point, not re-baselined to the
            # current log size.
            self.assertEqual(s.interrupted_idle_log_off, baseline_at_interrupt)

            # Now the watcher runs and must observe the post-interrupt activity.
            log_runtime.update_meta_counters()
            self.assertFalse(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, 0)
            self.assertTrue(s.interrupted_idle_suppressed)

            # Listing projection: non-idle log + cleared override -> busy True.
            coordinator = _list_coordinator(
                sessions=sessions,
                tmp_path=Path(td),
                log_runtime=log_runtime,
                idle_from_log_path=lambda _sid, _path: False,
            )
            out = coordinator.list_sessions()
            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], True)

    def test_discovery_refresh_broker_false_clears_suppression(self) -> None:
        # When discovery observes the broker has cleared interrupted_idle, the
        # refresh must clear the stored override AND its suppression so a later
        # interrupt can record a fresh baseline.
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            self._interrupted_turn_log(log_path)
            s = _make_session(log_path, interrupted_idle=True)
            sessions = {"broker-1": s}
            lock = threading.Lock()
            registry = _discovery_registry(sessions=sessions, lock=lock)

            # Stale override was already suppressed by the watcher.
            suppress_session_interrupted_idle(s)
            self.assertTrue(s.interrupted_idle_suppressed)

            size = int(log_path.stat().st_size)
            reg = _discovery_registration(
                session=s,
                log_path=log_path,
                interrupted_idle=False,
                meta_log_off=size,
            )
            registry.upsert_registration(reg)
            self.assertFalse(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, 0)
            self.assertFalse(s.interrupted_idle_suppressed)

    def test_discovery_refresh_suppressed_stale_true_stays_cleared(self) -> None:
        # After suppression, a discovery refresh still seeing the stale broker
        # True must not reactivate the override.
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            self._interrupted_turn_log(log_path)
            s = _make_session(log_path, interrupted_idle=True)
            sessions = {"broker-1": s}
            lock = threading.Lock()
            registry = _discovery_registry(sessions=sessions, lock=lock)

            suppress_session_interrupted_idle(s)
            size = int(log_path.stat().st_size)
            reg = _discovery_registration(
                session=s,
                log_path=log_path,
                interrupted_idle=True,
                meta_log_off=size,
            )
            registry.upsert_registration(reg)
            self.assertFalse(s.interrupted_idle)
            self.assertEqual(s.interrupted_idle_log_off, 0)
            self.assertTrue(s.interrupted_idle_suppressed)


class TestFreshDiscoveryPreservesInterruptBaseline(unittest.TestCase):
    """Fresh server discovery / empty registry: a brand-new registration that
    reports ``busy=False, interrupted_idle=True`` over a non-final log must store
    an active interrupted-idle baseline and list as idle. Before the fix the
    new-registration branch constructed the Session with the override, then
    called ``reset_log_caches`` which cleared it, so the listing projected
    ``busy=True`` against the broker's idle-interrupted report.
    """

    def _interrupted_turn_log(self, log_path: Path) -> None:
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

    def test_fresh_discovery_interrupted_idle_over_busy_log_stores_baseline_and_lists_idle(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            self._interrupted_turn_log(log_path)
            size = int(log_path.stat().st_size)
            self.assertGreater(size, 0)
            # Empty registry: this is a fresh server restart / first discovery.
            sessions: dict[str, Session] = {}
            lock = threading.Lock()
            log_runtime = _log_runtime(sessions)
            registry = _discovery_registry(sessions=sessions, lock=lock)

            template = _make_session(log_path, interrupted_idle=False)
            reg = _discovery_registration(
                session=template,
                log_path=log_path,
                interrupted_idle=True,
                meta_log_off=size,
            )
            registry.upsert_registration(reg)

            stored = sessions.get("broker-1")
            self.assertIsNotNone(stored)
            self.assertTrue(stored.interrupted_idle)
            # Active baseline recorded at current log size (not lost to reset).
            self.assertEqual(stored.interrupted_idle_log_off, size)
            self.assertFalse(stored.interrupted_idle_suppressed)

            # Listing must project busy=False despite a non-idle/non-final log,
            # because the override is alive.
            coordinator = _list_coordinator(
                sessions=sessions,
                tmp_path=Path(td),
                log_runtime=log_runtime,
                idle_from_log_path=lambda _sid, _path: False,
            )
            out = coordinator.list_sessions()
            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], False)
            self.assertTrue(stored.interrupted_idle)

    def test_fresh_discovery_false_interrupted_idle_keeps_clearing_semantics(self) -> None:
        # The false path must be unchanged: a fresh registration with
        # interrupted_idle=False stores no override and clears log caches.
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            self._interrupted_turn_log(log_path)
            size = int(log_path.stat().st_size)
            sessions: dict[str, Session] = {}
            lock = threading.Lock()
            log_runtime = _log_runtime(sessions)
            registry = _discovery_registry(sessions=sessions, lock=lock)

            template = _make_session(log_path, interrupted_idle=False)
            reg = _discovery_registration(
                session=template,
                log_path=log_path,
                interrupted_idle=False,
                meta_log_off=size,
            )
            registry.upsert_registration(reg)

            stored = sessions.get("broker-1")
            self.assertIsNotNone(stored)
            self.assertFalse(stored.interrupted_idle)
            self.assertEqual(stored.interrupted_idle_log_off, 0)
            self.assertFalse(stored.interrupted_idle_suppressed)
            # New-session clearing / log-cache semantics preserved.
            self.assertEqual(stored.meta_log_off, size)
            self.assertEqual(stored.idle_cache_log_off, -1)
            self.assertIsNone(stored.idle_cache_value)

            coordinator = _list_coordinator(
                sessions=sessions,
                tmp_path=Path(td),
                log_runtime=log_runtime,
                idle_from_log_path=lambda _sid, _path: False,
            )
            out = coordinator.list_sessions()
            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], True)


if __name__ == "__main__":
    unittest.main()
