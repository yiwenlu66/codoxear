"""Direct coordinator/impl tests for queue sweep + idle-guard behavior.

Previously these tests patched ``codoxear.server`` internals (~13 sites) and
built a ``SessionManager.__new__`` scaffold whose lambda-rebound methods
(``get_state`` / ``idle_from_log`` / ``send`` / ``_save_queues`` /
``_discover_existing_if_stale`` / ``_prune_dead_sessions`` /
``_maybe_drain_session_queue``) plus module-global patches of
``codoxear.server.time.time`` (8 sites), ``codoxear.server.QUEUE_SWEEP_MAX_DRAINS``
(2 sites) and ``codoxear.server.QUEUE_SWEEP_MAX_ATTEMPTS`` (2 sites) were the
real injection seams in disguise.

They now exercise the true seams directly:

* queue sweep orchestration (budget/cursor rotation over ready sessions) ->
  ``codoxear.queue_sweep.QueueSweepCoordinator`` built with the real
  ``codoxear.queue_store.QueueStore`` for orphan-recovery / drop-missing /
  nonempty-id bookkeeping, injected callables for the per-session drain
  boundary (``maybe_drain_session_queue``), filesystem persist
  (``save_queues``), and session discovery probes
  (``discover_existing_if_stale`` / ``prune_dead_sessions``), and constructor
  params for the former ``QUEUE_SWEEP_MAX_DRAINS`` / ``QUEUE_SWEEP_MAX_ATTEMPTS``
  globals.
* per-session queue drain + idle-grace guard ->
  ``codoxear.session_queue.SessionQueueCoordinator.promote_head_if_sendable``
  wired with the real ``codoxear.queue_runtime.queue_idle_grace_ready`` /
  ``start_queue_promotion`` / ``clear_queue_promotion`` runtime helpers and the
  real ``codoxear.queue_store.QueueStore`` promotion/commit-unknown/pop-sent
  state machine. ``now`` (time), ``remote_ready`` (broker readiness) and
  ``send`` (broker send) are injected callables.
* runtime-readiness authority (busy vs. log-idle vs. send-boundary ->
  remote_ready) -> the real ``codoxear.session_readiness.
  SessionReadinessCoordinator`` wired with the real
  ``codoxear.session_runtime.resolve_runtime_status`` /
  ``session_runtime_readiness`` model, the real
  ``codoxear.rollout_idle._compute_idle_from_log`` log parser (driven off real
  temp log files, no injected ``idle_from_log`` lambda), and the real
  ``codoxear.session_runtime.consume_session_confirmed_send_boundary`` /
  ``log_path_size_or_none`` send-boundary / log-size probes. The tests never
  recompute busy/idle themselves; the readiness coordinator owns it.

No ``codoxear.server.*`` module-global monkeypatching remains and no
``SessionManager`` is constructed. No ``patch`` calls remain: every former
patch target (``time.time``, ``QUEUE_SWEEP_MAX_DRAINS``,
``QUEUE_SWEEP_MAX_ATTEMPTS``) is now a constructor-injected callable/value or
a ``now_ts=`` argument on the coordinator method. No file under ``codoxear/``
is modified. No ``try/except`` swallows.
"""

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from codoxear.queue_sweep import QueueSweepCoordinator
from codoxear.queue_store import QueueStore
from codoxear.rollout_idle import _compute_idle_from_log
from codoxear.session_model import Session
from codoxear.session_queue import SessionQueueCoordinator
from codoxear.session_readiness import SessionReadinessCoordinator
from codoxear.session_runtime import consume_session_confirmed_send_boundary
from codoxear.session_runtime import log_path_size_or_none

# Server-configured defaults (codoxear/server_config.py): kept as module
# constants so the coordinators are wired with the same values the product uses.
QUEUE_IDLE_GRACE_SECONDS = 10.0
QUEUE_SWEEP_MAX_DRAINS = 4
QUEUE_SWEEP_MAX_ATTEMPTS = 16


class _NotReady(Exception):
    pass


class _InjectionError(Exception):
    pass


class _CommitUnknown(Exception):
    pass


def _make_session(
    *,
    sid: str,
    thread_id: str,
    log_path: Path | None,
    sock_path: Path,
    queue_idle_since: float | None = None,
    last_send_boundary_active: bool = False,
    last_send_log_path: Path | None = None,
    last_send_log_size: int | None = None,
) -> Session:
    return Session(
        session_id=sid,
        thread_id=thread_id,
        broker_pid=1,
        codex_pid=1,
        agent_backend="codex",
        owned=False,
        start_ts=0.0,
        cwd="/tmp",
        log_path=log_path,
        sock_path=sock_path,
        queue_idle_since=queue_idle_since,
        last_send_boundary_active=last_send_boundary_active,
        last_send_log_path=last_send_log_path,
        last_send_log_size=last_send_log_size,
    )


def _queue_item(item_id: str, text: str) -> dict[str, object]:
    return {"id": item_id, "text": text, "created_ts": 1.0}


class _QueueHarness:
    """Owns the mutable in-process state both coordinators close over and
    wires the real readiness + queue state machines with every external
    boundary injected.

    Boundaries injected (each replaces a former ``codoxear.server`` patch or
    lambda-rebound ``SessionManager`` method):

    * ``get_state`` / ``send``            -> broker control socket (process/socket/network).
    * ``now`` / ``now_ts``                -> wall clock (time).
    * ``save_queues``                     -> on-disk session_queues.json persist (filesystem).
    * ``input_lock_for_session``          -> per-session reentrancy guard (thread).
    * ``refresh_session_meta_if_sidecar_exists`` -> sidecar metadata probe (filesystem); inert here.
    * ``discover_existing_if_stale`` / ``prune_dead_sessions`` -> filesystem/process discovery probes;
      inert here because session membership is set up directly.

    Real (not injected): ``_compute_idle_from_log`` (log parser, driven off real
    temp log files), ``resolve_runtime_status`` / ``session_runtime_readiness``
    (busy/idle/boundary model), ``consume_session_confirmed_send_boundary`` /
    ``log_path_size_or_none`` (send-boundary + log-size probes),
    ``queue_idle_grace_ready`` / ``start_queue_promotion`` /
    ``clear_queue_promotion`` (idle-grace runtime), and the full ``QueueStore``
    append/promotion/commit-unknown/pop-sent/drop-missing/nonempty-id state
    machine.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.sessions: dict[str, Session] = {}
        self.queues: dict[str, list[dict[str, Any]]] = {}
        self.commit_unknown_sends: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._input_locks: dict[str, threading.RLock] = {}
        self.store = QueueStore(tmp_path / "queues.json")
        self.saves: list[int] = []

    def _input_lock(self, sid: str) -> threading.RLock:
        return self._input_locks.setdefault(sid, threading.RLock())

    def _boundary_unresolved(self, session_id: str, log_path: Path | None, log_size: int | None) -> bool:
        with self._lock:
            session = self.sessions.get(session_id)
            return consume_session_confirmed_send_boundary(session, log_path, log_size)

    def _idle_from_log(self, session_id: str) -> bool | None:
        # Real log parser over a real temp log file. No injected lambda.
        with self._lock:
            session = self.sessions.get(session_id)
            log_path = session.log_path if session is not None else None
        if not isinstance(log_path, Path) or not log_path.exists():
            return None
        return bool(_compute_idle_from_log(log_path))

    def readiness_coordinator(self, *, get_state) -> SessionReadinessCoordinator:
        return SessionReadinessCoordinator(
            lock=self._lock,
            sessions=lambda: self.sessions,
            refresh_session_meta_if_sidecar_exists=lambda *a, **k: None,
            get_state=get_state,
            log_size_or_none=log_path_size_or_none,
            confirmed_send_boundary_unresolved_for_session=self._boundary_unresolved,
            idle_from_log=self._idle_from_log,
            queue_len=lambda sid: self.store.queue_len(self.queues, sid),
            not_ready_error=_NotReady,
        )

    def queue_coordinator(
        self,
        *,
        get_state,
        send,
        now,
    ) -> SessionQueueCoordinator:
        readiness = self.readiness_coordinator(get_state=get_state)
        return SessionQueueCoordinator(
            lock=self._lock,
            sessions=lambda: self.sessions,
            queues=lambda: self.queues,
            queue_store=lambda: self.store,
            commit_unknown_sends=lambda: self.commit_unknown_sends,
            save_queues=lambda: self.saves.append(1),
            input_lock_for_session=self._input_lock,
            remote_ready=lambda session_id, log_path: readiness.queue_remote_ready(session_id, log_path=log_path),
            send=send,
            not_ready_error=_NotReady,
            retryable_send_errors=(_NotReady, _InjectionError),
            commit_unknown_error=_CommitUnknown,
            queue_idle_grace_seconds=QUEUE_IDLE_GRACE_SECONDS,
            now=now,
        )


class TestQueueSweepIdleGuard(unittest.TestCase):
    """Per-session idle-guard drain tests.

    Each case maps 1:1 to a former ``SessionManager._queue_sweep`` scenario,
    but drives ``SessionQueueCoordinator.promote_head_if_sendable`` (the sweep
    loop's per-session drain) directly. The sweep loop only calls this once
    per active queued session per tick, so one ``promote_head_if_sendable``
    call corresponds exactly to one former ``_queue_sweep`` tick for that
    session.
    """

    def test_queue_sweep_skips_when_log_not_idle(self) -> None:
        # Formerly: idle_from_log=False -> remote_ready False -> no send,
        # queue preserved, queue_idle_since reset to None.
        with TemporaryDirectory() as td:
            lp = Path(td) / "rollout.jsonl"
            lp.write_text(
                '{"type":"event_msg","payload":{"type":"user_message","message":"hi"},"timestamp":"2026-03-06T00:00:00Z"}\n',
                encoding="utf-8",
            )
            h = _QueueHarness(Path(td))
            h.sessions["s1"] = _make_session(sid="s1", thread_id="t1", log_path=lp, sock_path=lp.with_suffix(".sock"))
            h.queues["s1"] = [_queue_item("q1", "queued")]

            sent: list[tuple[str, str]] = []
            coord = h.queue_coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text, **kw: sent.append((sid, text)) or {"queued": False, "queue_len": 0},
                now=lambda: 100.0,
            )
            resp = coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=100.0)

            self.assertIsNone(resp)
            self.assertEqual(sent, [])
            self.assertEqual([item["text"] for item in h.queues["s1"]], ["queued"])
            self.assertIsNone(h.sessions["s1"].queue_idle_since)

    def test_queue_sweep_injects_when_log_idle(self) -> None:
        # Formerly: first sweep sets queue_idle_since; second sweep past grace
        # sends and pops the queued item.
        with TemporaryDirectory() as td:
            lp = Path(td) / "rollout.jsonl"
            lp.write_text(
                '{"type":"event_msg","payload":{"type":"task_complete"},"timestamp":"2026-03-06T00:00:00Z"}\n',
                encoding="utf-8",
            )
            h = _QueueHarness(Path(td))
            h.sessions["s1"] = _make_session(sid="s1", thread_id="t1", log_path=lp, sock_path=lp.with_suffix(".sock"))
            h.queues["s1"] = [_queue_item("q1", "queued")]

            sent: list[tuple[str, str]] = []
            coord = h.queue_coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text, **kw: sent.append((sid, text)) or {"queued": False, "queue_len": 0},
                now=lambda: 100.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1,
            )

            resp1 = coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=100.0)
            self.assertIsNone(resp1)
            self.assertEqual(sent, [])
            self.assertEqual(h.sessions["s1"].queue_idle_since, 100.0)

            resp2 = coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=100.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1)
            self.assertEqual(sent, [("s1", "queued")])
            self.assertNotContains("s1", h.queues)

    def test_queue_sweep_keeps_idle_window_when_log_idle_overrides_stale_broker_busy(self) -> None:
        # Formerly: log-idle=True keeps remote_ready True even when a stale
        # broker reports busy; idle window is not reset by a busy report, and
        # the grace gate still must elapse before sending.
        with TemporaryDirectory() as td:
            lp = Path(td) / "rollout.jsonl"
            lp.write_text(
                '{"type":"event_msg","payload":{"type":"task_complete"},"timestamp":"2026-03-06T00:00:00Z"}\n',
                encoding="utf-8",
            )
            h = _QueueHarness(Path(td))
            h.sessions["s1"] = _make_session(sid="s1", thread_id="t1", log_path=lp, sock_path=lp.with_suffix(".sock"))
            h.queues["s1"] = [_queue_item("q1", "queued")]

            busy_state = {"busy": False}

            def get_state(sid: str) -> dict:
                return {"busy": busy_state["busy"], "queue_len": 0}

            sent: list[tuple[str, str]] = []
            coord = h.queue_coordinator(
                get_state=get_state,
                send=lambda sid, text, **kw: sent.append((sid, text)) or {"queued": False, "queue_len": 0},
                now=lambda: 0.0,
            )

            # t=200: log idle -> remote_ready True; queue_idle_since None -> start window.
            self.assertIsNone(coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=200.0))
            self.assertEqual(sent, [])
            self.assertEqual(h.sessions["s1"].queue_idle_since, 200.0)

            # Stale broker now reports busy; log-idle override keeps remote_ready
            # True, so the idle window is NOT reset.
            busy_state["busy"] = True
            self.assertIsNone(coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=204.0))
            self.assertEqual(sent, [])
            self.assertEqual(h.sessions["s1"].queue_idle_since, 200.0)

            # Still within grace.
            self.assertIsNone(coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=200.0 + QUEUE_IDLE_GRACE_SECONDS - 0.1))
            self.assertEqual(sent, [])

            # Past grace -> send.
            resp = coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=200.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1)
            self.assertEqual(sent, [("s1", "queued")])

    def test_queue_sweep_blocks_broker_busy_until_log_advances_after_send(self) -> None:
        # Formerly: an unresolved confirmed-send boundary (last_send_log_size
        # not yet advanced) keeps remote_ready False; once the log grows past
        # the boundary it resolves, log-idle wins, and the idle window starts.
        with TemporaryDirectory() as td:
            lp = Path(td) / "rollout.jsonl"
            lp.write_text(
                '{"type":"event_msg","payload":{"type":"task_complete"},"timestamp":"2026-03-06T00:00:00Z"}\n',
                encoding="utf-8",
            )
            boundary_size = lp.stat().st_size
            h = _QueueHarness(Path(td))
            h.sessions["s1"] = _make_session(
                sid="s1",
                thread_id="t1",
                log_path=lp,
                sock_path=lp.with_suffix(".sock"),
                last_send_boundary_active=True,
                last_send_log_path=lp,
                last_send_log_size=boundary_size,
            )
            h.queues["s1"] = [_queue_item("q1", "queued")]

            sent: list[tuple[str, str]] = []
            coord = h.queue_coordinator(
                get_state=lambda sid: {"busy": True, "queue_len": 0},
                send=lambda sid, text, **kw: sent.append((sid, text)) or {"queued": False, "queue_len": 0},
                now=lambda: 201.0,
            )

            # Boundary unresolved (log has not advanced) -> remote_ready False.
            self.assertIsNone(coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=200.0))
            self.assertEqual(sent, [])
            self.assertIsNone(h.sessions["s1"].queue_idle_since)

            # Log advances past the send boundary -> boundary resolves & clears.
            lp.write_text(
                lp.read_text(encoding="utf-8")
                + '{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n',
                encoding="utf-8",
            )
            self.assertIsNone(coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=201.0))
            self.assertEqual(sent, [])
            self.assertEqual(h.sessions["s1"].queue_idle_since, 201.0)
            # Boundary was consumed once resolved.
            self.assertFalse(h.sessions["s1"].last_send_boundary_active)

    def test_queue_sweep_pops_duplicate_texts_by_item_id(self) -> None:
        # Formerly: two items with identical text -> only the head is sent and
        # popped; the second item remains by id.
        with TemporaryDirectory() as td:
            lp = Path(td) / "rollout.jsonl"
            lp.write_text(
                '{"type":"event_msg","payload":{"type":"task_complete"},"timestamp":"2026-03-06T00:00:00Z"}\n',
                encoding="utf-8",
            )
            h = _QueueHarness(Path(td))
            h.sessions["s1"] = _make_session(sid="s1", thread_id="t1", log_path=lp, sock_path=lp.with_suffix(".sock"))
            h.queues["s1"] = [_queue_item("q1", "dup"), _queue_item("q2", "dup")]

            sent: list[tuple[str, str]] = []
            coord = h.queue_coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text, **kw: sent.append((sid, text)) or {"queued": False, "queue_len": 0},
                now=lambda: 300.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1,
            )

            self.assertIsNone(coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=300.0))
            resp = coord.promote_head_if_sendable("s1", require_idle_grace=True, now_ts=300.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1)

            self.assertEqual(sent, [("s1", "dup")])
            self.assertEqual([item["id"] for item in h.queues["s1"]], ["q2"])


class TestQueueSweepOrchestration(unittest.TestCase):
    """Sweep-loop budget/cursor tests -> ``QueueSweepCoordinator.sweep``.

    The per-session drain boundary (``maybe_drain_session_queue``) is injected:
    for the multi-ready-session budget case it delegates to the real
    ``SessionQueueCoordinator.promote_head_if_sendable`` (so the idle-guard
    + send path is exercised end-to-end), and for the attempt-rotation case it
    is a pure predicate (the focus is the orchestration, not the drain).
    """

    @staticmethod
    def _sweep_coordinator(
        harness: _QueueHarness,
        *,
        maybe_drain_session_queue,
        max_drains_per_sweep: int = QUEUE_SWEEP_MAX_DRAINS,
        max_attempts_per_sweep: int | None = QUEUE_SWEEP_MAX_ATTEMPTS,
    ) -> tuple[QueueSweepCoordinator, list[int]]:
        cursor_box: list[int] = [0]

        def queue_sweep_cursor() -> int:
            return cursor_box[0]

        def set_queue_sweep_cursor(value: int) -> None:
            cursor_box[0] = int(value)

        coord = QueueSweepCoordinator(
            lock=harness._lock,
            sessions=lambda: harness.sessions,
            queues=lambda: harness.queues,
            commit_unknown_sends=lambda: harness.commit_unknown_sends,
            queue_store=harness.store,
            discover_existing_if_stale=lambda: None,
            prune_dead_sessions=lambda: None,
            mark_queue_orphan_recovery_locked=lambda sid: harness.store.mark_orphan_recovery_items(harness.queues, sid),
            save_queues=lambda: harness.saves.append(1),
            maybe_drain_session_queue=maybe_drain_session_queue,
            max_drains_per_sweep=max_drains_per_sweep,
            max_attempts_per_sweep=max_attempts_per_sweep,
            queue_sweep_cursor=queue_sweep_cursor,
            set_queue_sweep_cursor=set_queue_sweep_cursor,
        )
        return coord, cursor_box

    def test_queue_sweep_drains_multiple_ready_sessions_up_to_budget(self) -> None:
        # Formerly: three idle+ready sessions, QUEUE_SWEEP_MAX_DRAINS=2 -> only
        # the first two drain in one sweep, cursor advances to 2, third queue
        # is preserved. The drain delegates to the real idle-guard path.
        with TemporaryDirectory() as td:
            h = _QueueHarness(Path(td))
            sent: list[tuple[str, str]] = []
            for idx, sid in enumerate(["s1", "s2", "s3"], start=1):
                lp = Path(td) / f"rollout-{sid}.jsonl"
                lp.write_text(
                    '{"type":"event_msg","payload":{"type":"task_complete"},"timestamp":"2026-03-06T00:00:00Z"}\n',
                    encoding="utf-8",
                )
                h.sessions[sid] = _make_session(
                    sid=sid,
                    thread_id=f"t{idx}",
                    log_path=lp,
                    sock_path=lp.with_suffix(".sock"),
                    queue_idle_since=10.0,
                )
                h.queues[sid] = [_queue_item(f"q{idx}", f"queued-{idx}")]

            queue_coord = h.queue_coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text, **kw: sent.append((sid, text)) or {"queued": False, "queue_len": 0},
                now=lambda: 10.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1,
            )

            def maybe_drain(session_id: str) -> bool:
                return bool(
                    queue_coord.promote_head_if_sendable(
                        session_id,
                        require_idle_grace=True,
                        now_ts=10.0 + QUEUE_IDLE_GRACE_SECONDS + 0.1,
                    )
                )

            sweep_coord, cursor_box = self._sweep_coordinator(
                h,
                maybe_drain_session_queue=maybe_drain,
                max_drains_per_sweep=2,
                max_attempts_per_sweep=QUEUE_SWEEP_MAX_ATTEMPTS,
            )
            sweep_coord.sweep()

            self.assertEqual(sent, [("s1", "queued-1"), ("s2", "queued-2")])
            self.assertNotContains("s1", h.queues)
            self.assertNotContains("s2", h.queues)
            self.assertEqual(cursor_box[0], 2)
            self.assertEqual([item["text"] for item in h.queues["s3"]], ["queued-3"])

    def test_queue_sweep_attempt_budget_rotates_past_unready_prefix(self) -> None:
        # Formerly: QUEUE_SWEEP_MAX_DRAINS=1, QUEUE_SWEEP_MAX_ATTEMPTS=2, only
        # "s3" drainable -> first sweep attempts s1,s2 (cursor->2), second
        # sweep attempts s3 and drains it (cursor->3). The drain is an injected
        # predicate because this case targets the attempt budget / cursor
        # rotation, not the idle guard.
        with TemporaryDirectory() as td:
            h = _QueueHarness(Path(td))
            for idx, sid in enumerate(["s1", "s2", "s3", "s4"], start=1):
                h.sessions[sid] = _make_session(
                    sid=sid,
                    thread_id=f"t{idx}",
                    log_path=None,
                    sock_path=Path(td) / f"{sid}.sock",
                )
                h.queues[sid] = [_queue_item(f"q{idx}", f"queued-{idx}")]

            attempts: list[str] = []

            def maybe_drain(session_id: str) -> bool:
                attempts.append(session_id)
                return session_id == "s3"

            sweep_coord, cursor_box = self._sweep_coordinator(
                h,
                maybe_drain_session_queue=maybe_drain,
                max_drains_per_sweep=1,
                max_attempts_per_sweep=2,
            )

            sweep_coord.sweep()
            self.assertEqual(attempts, ["s1", "s2"])
            self.assertEqual(cursor_box[0], 2)

            attempts.clear()
            sweep_coord.sweep()
            self.assertEqual(attempts, ["s3"])
            self.assertEqual(cursor_box[0], 3)


if __name__ == "__main__":
    unittest.main()
