"""Direct coordinator/dependency-injection tests for unattended sweep logic.

Previously these tests patched ``codoxear.server`` internals (~17 sites):
``codoxear.server.time.time``, ``codoxear.server._last_chat_role_ts_from_tail``,
``codoxear.server.UNATTENDED_PROMPT_PREFIX`` plus a ``SessionManager.__new__``
scaffold whose lambda-rebound ``get_state`` / ``send`` / ``_save_unattended``
methods were the real injection seams in disguise.

They now exercise the true seams directly:

* unattended sweep orchestration -> ``codoxear.unattended_sweep.
  UnattendedSweepCoordinator`` built with the real ``codoxear.unattended``
  decision helpers and injected callables for the broker-socket (``get_state``,
  ``send``), log-tail (``last_chat_role_ts_from_tail``), time (``now``), and
  filesystem (``save_unattended``) boundaries.
* unattended config get/set masking -> ``codoxear.session_unattended_config.
  SessionUnattendedConfigCoordinator`` with the real
  ``clean_unattended_cooldown_minutes`` / ``clean_unattended_remaining_injections``
  cleaners injected.
* runtime readiness gate -> ``codoxear.session_readiness.
  SessionReadinessCoordinator.runtime_status_from_state_and_log`` wired with the
  real ``broker_runtime_state`` / ``resolve_runtime_status`` /
  ``session_runtime_readiness`` model and ``_compute_idle_from_log`` log parsing;
  only the broker send-boundary probe (``confirmed_send_boundary_unresolved``)
  is injected inert, since these unit tests never exercised a live broker socket.

No ``codoxear.server.*`` module-global monkeypatching remains and no
``SessionManager`` is constructed. No ``patch`` calls remain: every former patch
target (``time.time``, ``_last_chat_role_ts_from_tail``, ``UNATTENDED_PROMPT_PREFIX``)
is now a constructor-injected callable/value on the coordinator. No file under
``codoxear/`` is modified. No ``try/except`` swallows.
"""

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.rollout_idle import _compute_idle_from_log
from codoxear.rollout_idle import _last_chat_role_ts_from_tail as _last_chat_role_ts_from_tail_impl
from codoxear.session_errors import SessionCommitUnknownError
from codoxear.session_model import Session
from codoxear.session_readiness import SessionReadinessCoordinator
from codoxear.session_unattended_config import SessionUnattendedConfigCoordinator
from codoxear.unattended import UNATTENDED_PROMPT_PREFIX
from codoxear.unattended import clean_unattended_cooldown_minutes as _clean_cooldown_impl
from codoxear.unattended import clean_unattended_remaining_injections as _clean_remaining_impl
from codoxear.unattended_sweep import UnattendedSweepCoordinator

# Server-configured defaults (codoxear/server_config.py): kept as module
# constants so the coordinators are wired with the same values the product uses.
DEFAULT_IDLE_MINUTES = 5
DEFAULT_MAX_INJECTIONS = 10
MAX_SCAN_BYTES = 8 * 1024 * 1024


def _clean_cooldown(raw) -> int:
    return _clean_cooldown_impl(raw, default_idle_minutes=DEFAULT_IDLE_MINUTES)


def _clean_remaining(raw, *, allow_zero: bool) -> int:
    return _clean_remaining_impl(raw, default_max_injections=DEFAULT_MAX_INJECTIONS, allow_zero=allow_zero)


def _make_session(*, sid: str, thread_id: str, log_path: Path) -> Session:
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
        sock_path=log_path.with_suffix(".sock"),
    )


class _SweepHarness:
    """Owns the mutable in-process state the sweep coordinator closes over and
    builds the coordinator with every external boundary injected.

    Boundaries injected (each replaces a former ``codoxear.server`` patch):

    * ``get_state`` / ``send``      -> broker control socket (process/socket/network).
    * ``last_chat_role_ts_from_tail`` -> backend log tail reader (filesystem read).
    * ``now``                         -> wall clock (time).
    * ``save_unattended``             -> on-disk unattended.json persist (filesystem).
    * ``discover_existing_if_stale`` / ``prune_dead_sessions`` -> filesystem/process
      discovery probes; inert here because session membership is set up directly.
    * ``input_lock_for_session``      -> per-session reentrancy guard (thread).
    * ``runtime_status_from_state``   -> readiness model; real
      ``SessionReadinessCoordinator`` with the broker send-boundary probe inert.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.unattended: dict[str, dict] = {}
        self.unattended_last_injected: dict[str, float] = {}
        self.unattended_last_injected_scope: dict[str, float] = {}
        self._lock = threading.Lock()
        self._input_locks: dict[str, threading.Lock] = {}
        self.saves: list[bool] = []

    def _input_lock(self, sid: str) -> threading.Lock:
        return self._input_locks.setdefault(sid, threading.Lock())

    def _runtime_status(self, session_id: str, state: dict, log_path: Path | None):
        # Real readiness model wired with real log parsing; only the broker
        # send-boundary probe is inert (these tests never touch a live socket).
        def idle_from_log(sid: str):
            session = self.sessions.get(sid)
            if session is None or session.log_path is None or not session.log_path.exists():
                return None
            return bool(_compute_idle_from_log(session.log_path))

        readiness = SessionReadinessCoordinator(
            lock=self._lock,
            sessions=lambda: self.sessions,
            refresh_session_meta_if_sidecar_exists=lambda *a, **k: None,
            get_state=lambda sid: {"busy": False, "queue_len": 0},
            log_size_or_none=lambda lp: (int(lp.stat().st_size) if (lp is not None and Path(lp).exists()) else None),
            confirmed_send_boundary_unresolved_for_session=lambda sid, lp, sz: False,
            idle_from_log=idle_from_log,
            queue_len=lambda sid: 0,
            not_ready_error=RuntimeError,
        )
        return readiness.runtime_status_from_state_and_log(session_id, state, log_path)

    def coordinator(
        self,
        *,
        get_state,
        send,
        last_chat_role_ts_from_tail=_last_chat_role_ts_from_tail_impl,
        now: "callable" = lambda: 0.0,
        prompt_prefix: str = UNATTENDED_PROMPT_PREFIX,
        record_saves: bool = False,
    ) -> UnattendedSweepCoordinator:
        return UnattendedSweepCoordinator(
            lock=self._lock,
            sessions=lambda: self.sessions,
            unattended=lambda: self.unattended,
            unattended_last_injected=lambda: self.unattended_last_injected,
            unattended_last_injected_scope=lambda: self.unattended_last_injected_scope,
            discover_existing_if_stale=lambda: None,
            prune_dead_sessions=lambda: None,
            input_lock_for_session=self._input_lock,
            save_unattended=((lambda: self.saves.append(True)) if record_saves else (lambda: None)),
            get_state=get_state,
            runtime_status_from_state=self._runtime_status,
            queue_len=lambda sid: 0,
            last_chat_role_ts_from_tail=last_chat_role_ts_from_tail,
            send=send,
            now=now,
            prompt_prefix=prompt_prefix,
            default_idle_minutes=DEFAULT_IDLE_MINUTES,
            default_max_injections=DEFAULT_MAX_INJECTIONS,
            max_scan_bytes=MAX_SCAN_BYTES,
        )

    def config_coordinator(self, *, save_unattended=None) -> SessionUnattendedConfigCoordinator:
        return SessionUnattendedConfigCoordinator(
            lock=self._lock,
            sessions=lambda: self.sessions,
            unattended=lambda: self.unattended,
            unattended_last_injected=lambda: self.unattended_last_injected,
            input_lock_for_session=self._input_lock,
            save_unattended=save_unattended or (lambda: None),
            clean_unattended_cooldown_minutes=_clean_cooldown,
            clean_unattended_remaining_injections=_clean_remaining,
        )


class TestUnattendedSweep(unittest.TestCase):
    def test_unattended_set_never_stores_enabled_with_zero_remaining(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": False, "request": "A", "cooldown_minutes": 5, "remaining_injections": 0}
            h.unattended_last_injected["sid-a"] = 1000.0
            cfg_coord = h.config_coordinator()

            cfg = cfg_coord.set("sid-a", enabled=True)

            self.assertFalse(cfg["enabled"])
            self.assertFalse(h.unattended["sid-a"]["enabled"])
            self.assertEqual(h.unattended["sid-a"]["remaining_injections"], 0)
            self.assertNotIn("sid-a", h.unattended_last_injected)

    def test_unattended_partial_request_save_preserves_server_budget_decrement(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": False, "request": "old", "cooldown_minutes": 5, "remaining_injections": 0}
            cfg_coord = h.config_coordinator()

            cfg = cfg_coord.set("sid-a", request="new")

            self.assertEqual(cfg["request"], "new")
            self.assertFalse(cfg["enabled"])
            self.assertEqual(cfg["remaining_injections"], 0)
            self.assertEqual(h.unattended["sid-a"]["remaining_injections"], 0)

    def test_unattended_get_masks_stale_enabled_zero_remaining(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 0}
            cfg_coord = h.config_coordinator()

            cfg = cfg_coord.get("sid-a")

            self.assertFalse(cfg["enabled"])
            self.assertEqual(cfg["remaining_injections"], 0)

    def test_does_not_inject_after_non_final_assistant_narration(self) -> None:
        # Real log-tail reader (no injection): an analysis-phase agent_message is
        # not a final turn end, so the tail yields no injectable assistant ts.
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"type":"event_msg","ts":1,"payload":{"type":"user_message","message":"start"}}',
                        '{"type":"event_msg","ts":600,"payload":{"type":"agent_message","phase":"analysis","message":"still working"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 10}
            sent: list[tuple[str, str]] = []

            coord = h.coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text: (sent.append((sid, text)) or {"ok": True}),
                now=lambda: 1000.0,
            )
            coord.sweep()

            self.assertEqual(sent, [])
            self.assertEqual(h.unattended["sid-a"]["remaining_injections"], 10)

    def test_rechecks_latest_assistant_timestamp_before_send(self) -> None:
        # Injected tail reader with a side-effect sequence: the pre-probe tail
        # is old enough to pass the cooldown gate, but the post-probe recheck
        # sees a fresh assistant ts within cooldown -> injection is suppressed.
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 10}
            sent: list[tuple[str, str]] = []
            tail_seq = iter([("assistant", 600.0), ("assistant", 950.0)])

            def tail(*_a, **_k):
                return next(tail_seq)

            coord = h.coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text: (sent.append((sid, text)) or {"ok": True}),
                last_chat_role_ts_from_tail=tail,
                now=lambda: 1000.0,
            )
            coord.sweep()

            self.assertEqual(sent, [])
            self.assertEqual(h.unattended["sid-a"]["remaining_injections"], 10)

    def test_rechecks_config_after_idle_probe_before_send(self) -> None:
        # The coordinator re-reads the live config under the input lock before
        # sending; disabling it during the idle probe must suppress the send and
        # preserve the freshly-disabled config.
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": True, "request": "old", "cooldown_minutes": 5, "remaining_injections": 10}
            sent: list[tuple[str, str]] = []

            def disable_during_idle_probe(sid: str) -> dict:
                h.unattended[sid] = {"enabled": False, "request": "new", "cooldown_minutes": 5, "remaining_injections": 0}
                return {"busy": False, "queue_len": 0}

            coord = h.coordinator(
                get_state=disable_during_idle_probe,
                send=lambda sid, text: (sent.append((sid, text)) or {"ok": True}),
                last_chat_role_ts_from_tail=lambda *_a, **_k: ("assistant", 0.0),
                now=lambda: 1000.0,
            )
            coord.sweep()

            self.assertEqual(sent, [])
            self.assertFalse(h.unattended["sid-a"]["enabled"])
            self.assertEqual(h.unattended["sid-a"]["remaining_injections"], 0)

    def test_dedupes_injection_for_same_thread(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.sessions["sid-b"] = _make_session(sid="sid-b", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 10}
            h.unattended["sid-b"] = {"enabled": True, "request": "B", "cooldown_minutes": 5, "remaining_injections": 10}

            sent: list[tuple[str, str]] = []
            coord = h.coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text: (sent.append((sid, text)) or {"ok": True}),
                last_chat_role_ts_from_tail=lambda *_a, **_k: ("assistant", 600.0),
                now=lambda: 1000.0,
                prompt_prefix="PFX",
            )
            coord.sweep()

            self.assertEqual(sent, [("sid-a", "PFX\n\n---\n\nAdditional request from user: A\n")])
            self.assertIn("thread:thread-1", h.unattended_last_injected_scope)
            self.assertEqual(h.unattended["sid-a"]["remaining_injections"], 9)

    def test_injects_once_per_distinct_thread(self) -> None:
        with TemporaryDirectory() as td:
            p1 = Path(td) / "rollout-a.jsonl"
            p2 = Path(td) / "rollout-b.jsonl"
            p1.write_text("{}", encoding="utf-8")
            p2.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p1)
            h.sessions["sid-b"] = _make_session(sid="sid-b", thread_id="thread-2", log_path=p2)
            h.unattended["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 10}
            h.unattended["sid-b"] = {"enabled": True, "request": "B", "cooldown_minutes": 5, "remaining_injections": 10}

            sent: list[tuple[str, str]] = []
            coord = h.coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text: (sent.append((sid, text)) or {"ok": True}),
                last_chat_role_ts_from_tail=lambda *_a, **_k: ("assistant", 600.0),
                now=lambda: 1000.0,
                prompt_prefix="PFX",
            )
            coord.sweep()

            self.assertEqual(
                sent,
                [
                    ("sid-a", "PFX\n\n---\n\nAdditional request from user: A\n"),
                    ("sid-b", "PFX\n\n---\n\nAdditional request from user: B\n"),
                ],
            )

    def test_send_commit_unknown_preserves_budget_and_cooldown_state(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 3}
            attempts: list[tuple[str, str]] = []

            def commit_unknown_send(sid: str, text: str) -> dict:
                attempts.append((sid, text))
                raise SessionCommitUnknownError("send commit status unknown; broker did not reply before timeout")

            coord = h.coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=commit_unknown_send,
                last_chat_role_ts_from_tail=lambda *_a, **_k: ("assistant", 600.0),
                now=lambda: 1000.0,
                record_saves=True,
            )
            coord.sweep()

            self.assertEqual(len(attempts), 1)
            self.assertEqual(h.unattended["sid-a"]["remaining_injections"], 3)
            self.assertTrue(h.unattended["sid-a"]["enabled"])
            self.assertNotIn("sid-a", h.unattended_last_injected)
            self.assertNotIn("thread:thread-1", h.unattended_last_injected_scope)
            self.assertEqual(h.saves, [])

    def test_session_timeout_does_not_kill_other_injections(self) -> None:
        with TemporaryDirectory() as td:
            p1 = Path(td) / "rollout-a.jsonl"
            p2 = Path(td) / "rollout-b.jsonl"
            p1.write_text("{}", encoding="utf-8")
            p2.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-timeout"] = _make_session(sid="sid-timeout", thread_id="thread-timeout", log_path=p1)
            h.sessions["sid-ok"] = _make_session(sid="sid-ok", thread_id="thread-ok", log_path=p2)
            h.unattended["sid-timeout"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 10}
            h.unattended["sid-ok"] = {"enabled": True, "request": "B", "cooldown_minutes": 5, "remaining_injections": 10}

            sent: list[tuple[str, str]] = []

            def _state(sid: str) -> dict:
                if sid == "sid-timeout":
                    raise TimeoutError("timed out")
                return {"busy": False, "queue_len": 0}

            coord = h.coordinator(
                get_state=_state,
                send=lambda sid, text: (sent.append((sid, text)) or {"ok": True}),
                last_chat_role_ts_from_tail=lambda *_a, **_k: ("assistant", 600.0),
                now=lambda: 1000.0,
                prompt_prefix="PFX",
            )
            coord.sweep()

            self.assertEqual(sent, [("sid-ok", "PFX\n\n---\n\nAdditional request from user: B\n")])

    def test_uses_per_session_cooldown_minutes(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 2, "remaining_injections": 10}
            h.unattended_last_injected["sid-a"] = 950.0

            sent: list[tuple[str, str]] = []
            coord = h.coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text: (sent.append((sid, text)) or {"ok": True}),
                last_chat_role_ts_from_tail=lambda *_a, **_k: ("assistant", 600.0),
                now=lambda: 1000.0,
                prompt_prefix="PFX",
            )
            coord.sweep()

            self.assertEqual(sent, [])

    def test_dedupes_injection_for_three_sessions_sharing_thread(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            for sid, request in (("sid-a", "A"), ("sid-b", "B"), ("sid-c", "C")):
                h.sessions[sid] = _make_session(sid=sid, thread_id="thread-1", log_path=p)
                h.unattended[sid] = {"enabled": True, "request": request, "cooldown_minutes": 5, "remaining_injections": 10}

            sent: list[tuple[str, str]] = []
            coord = h.coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text: (sent.append((sid, text)) or {"ok": True}),
                last_chat_role_ts_from_tail=lambda *_a, **_k: ("assistant", 600.0),
                now=lambda: 1000.0,
                prompt_prefix="PFX",
            )
            coord.sweep()

            self.assertEqual(sent, [("sid-a", "PFX\n\n---\n\nAdditional request from user: A\n")])
            self.assertEqual(h.unattended["sid-a"]["remaining_injections"], 9)
            self.assertEqual(h.unattended["sid-b"]["remaining_injections"], 10)
            self.assertEqual(h.unattended["sid-c"]["remaining_injections"], 10)

    def test_zero_remaining_disables_without_sending(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 0}
            h.unattended_last_injected["sid-a"] = 900.0

            sent: list[tuple[str, str]] = []
            coord = h.coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text: (sent.append((sid, text)) or {"ok": True}),
                last_chat_role_ts_from_tail=lambda *_a, **_k: ("assistant", 600.0),
                now=lambda: 1000.0,
            )
            coord.sweep()

            self.assertEqual(sent, [])
            self.assertEqual(h.unattended["sid-a"]["remaining_injections"], 0)
            self.assertFalse(h.unattended["sid-a"]["enabled"])
            self.assertNotIn("sid-a", h.unattended_last_injected)

    def test_disables_unattended_after_last_injection(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "rollout.jsonl"
            p.write_text("{}", encoding="utf-8")

            h = _SweepHarness()
            h.sessions["sid-a"] = _make_session(sid="sid-a", thread_id="thread-1", log_path=p)
            h.unattended["sid-a"] = {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 1}

            sent: list[tuple[str, str]] = []
            coord = h.coordinator(
                get_state=lambda sid: {"busy": False, "queue_len": 0},
                send=lambda sid, text: (sent.append((sid, text)) or {"ok": True}),
                last_chat_role_ts_from_tail=lambda *_a, **_k: ("assistant", 600.0),
                now=lambda: 1000.0,
                prompt_prefix="PFX",
            )
            coord.sweep()

            self.assertEqual(sent, [("sid-a", "PFX\n\n---\n\nAdditional request from user: A\n")])
            self.assertEqual(h.unattended["sid-a"]["remaining_injections"], 0)
            self.assertFalse(h.unattended["sid-a"]["enabled"])


if __name__ == "__main__":
    unittest.main()
