"""Direct coordinator/route-deps tests for session pending/log-idle behavior.

These tests previously patched ``codoxear.server.MANAGER`` (~20 sites) and
constructed ``SessionManager`` via ``__new__`` with monkeypatched internals.
They now exercise the true seams directly:

* listing ``busy``/idle/send-boundary resolution -> ``SessionListCoordinator``
  (via ``ListingRuntimeProbes`` injection), matching ``tests/test_session_list.py``.
* the per-message runtime snapshot -> ``ServerRouteDepsFactory.message_runtime_snapshot``
  with a fake ``MANAGER`` injected through ``ServerRouteCaps`` (no module-global
  patch), backed by the real ``SessionReadinessCoordinator``.
* ``idle_from_log_path`` cache behavior -> ``SessionLogRuntimeCoordinator``.
* ``refresh_session_meta`` -> ``SessionRefreshCoordinator``.
* ``get_state`` -> ``SessionControlCoordinator``.
* ``_refresh_session_state`` -> ``SessionPruneCoordinator``.

No ``codoxear.server.MANAGER`` patch and no ``SessionManager.__new__`` construction
remain. No file under ``codoxear/`` is modified.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.server_route_deps import ServerRouteCaps
from codoxear.server_route_deps import ServerRouteDepsFactory
from codoxear.session_control import SessionControlCoordinator
from codoxear.session_list import SessionListCoordinator
from codoxear.session_log_runtime import SessionLogRuntimeCoordinator
from codoxear.session_model import Session
from codoxear.session_prune import SessionPruneCoordinator
from codoxear.session_readiness import SessionReadinessCoordinator
from codoxear.session_refresh import SessionRefreshCoordinator
from codoxear.session_runtime import ListingRuntimeProbes
from codoxear.session_runtime import broker_busy_queue
from codoxear.session_runtime import broker_interrupted_idle
from codoxear.session_runtime import consume_session_confirmed_send_boundary
from codoxear.session_runtime import log_path_size_or_none
from codoxear.session_runtime import reset_session_log_caches
from codoxear.session_runtime import select_runtime_token
from codoxear.session_store import SessionStore
from codoxear.session_store import SessionStorePaths
from codoxear.sidecar_metadata import _clean_optional_text as _sidecar_clean_optional_text


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _store(tmp_path: Path) -> SessionStore:
    return SessionStore(
        paths=SessionStorePaths(
            aliases=tmp_path / "aliases.json",
            sidebar_meta=tmp_path / "sidebar.json",
            hidden_sessions=tmp_path / "hidden.json",
            files=tmp_path / "files.json",
            queues=tmp_path / "queues.json",
            pending_attachments=tmp_path / "pending.json",
            commit_unknown_sends=tmp_path / "commit.json",
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


def _session(
    *,
    session_id: str = "broker-1",
    thread_id: str = "broker-1",
    agent_backend: str = "codex",
    log_path: Path | None = None,
    sock_path: Path | None = None,
    busy: bool = False,
    queue_len: int = 0,
    interrupted_idle: bool = False,
    token: dict | None = None,
    last_send_boundary_active: bool = False,
    last_send_log_path: Path | None = None,
    last_send_log_size: int | None = None,
) -> Session:
    return Session(
        session_id=session_id,
        thread_id=thread_id,
        broker_pid=1,
        codex_pid=2,
        agent_backend=agent_backend,
        owned=False,
        start_ts=123.0,
        cwd="/tmp",
        log_path=log_path,
        sock_path=sock_path or Path("/tmp/broker-1.sock"),
        busy=busy,
        queue_len=queue_len,
        interrupted_idle=interrupted_idle,
        token=token,
        last_send_boundary_active=last_send_boundary_active,
        last_send_log_path=last_send_log_path,
        last_send_log_size=last_send_log_size,
    )


def _probes(
    *,
    idle_from_log_path=lambda _sid, _path: True,
    send_boundary_unresolved=lambda _sid, _lp, _ls: False,
    log_size_or_none=log_path_size_or_none,
    current_git_branch=lambda _path: "main",
) -> ListingRuntimeProbes:
    return ListingRuntimeProbes(
        last_conversation_ts_from_tail=lambda _path: None,
        read_run_settings_from_log=lambda _path, _backend: (None, None, None),
        log_size_or_none=log_size_or_none,
        send_boundary_unresolved=send_boundary_unresolved,
        idle_from_log_path=idle_from_log_path,
        current_git_branch=current_git_branch,
    )


def _list_coordinator(
    *,
    sessions: dict[str, Session],
    tmp_path: Path,
    probes: ListingRuntimeProbes,
    queues: dict | None = None,
    unattended: dict | None = None,
    aliases: dict | None = None,
    store: SessionStore | None = None,
    now_ts: float = 200.0,
) -> SessionListCoordinator:
    store = store or _store(tmp_path)
    return SessionListCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        queues=lambda: queues if queues is not None else {},
        unattended=lambda: unattended if unattended is not None else {},
        aliases=lambda: aliases if aliases is not None else {},
        hidden_sessions=lambda: set(),
        commit_unknown_sends=lambda: {},
        store=store,
        discover_existing_if_stale=lambda: None,
        prune_dead_sessions=lambda: None,
        update_meta_counters=lambda: None,
        save_files=lambda: None,
        save_sidebar_meta=lambda: None,
        save_recent_cwds=lambda: None,
        now=lambda: now_ts,
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


def _caps(manager: object, *, select_runtime_token=select_runtime_token, **overrides) -> ServerRouteCaps:
    """Build a ServerRouteCaps with all fields defaulted to None except the two
    that ``message_runtime_snapshot`` reads (``MANAGER`` and
    ``_select_runtime_token``), plus any explicit overrides."""
    field_defaults = {f.name: None for f in dataclasses.fields(ServerRouteCaps)}
    field_defaults["MANAGER"] = manager
    field_defaults["_select_runtime_token"] = select_runtime_token
    field_defaults.update(overrides)
    return ServerRouteCaps(**field_defaults)


class _SnapshotManager:
    """Fake manager for ``ServerRouteDepsFactory.message_runtime_snapshot``.

    Provides exactly the three methods the factory calls (``get_state``,
    ``_runtime_status_from_state_and_log``, ``_queue_len``) and delegates the
    runtime-status resolution to the real ``SessionReadinessCoordinator`` so the
    genuine idle/boundary/token logic is exercised.
    """

    def __init__(
        self,
        *,
        session: Session | None,
        state: dict,
        idle: bool = False,
        queue_len: int = 0,
        sessions_mapping: dict[str, Session] | None = None,
    ) -> None:
        self._session = session
        self._state = state
        self._idle_fn = idle if callable(idle) else (lambda _sid: idle)
        self._queue_len_val = queue_len
        mapping = sessions_mapping if sessions_mapping is not None else (
            {session.session_id: session} if session is not None else {}
        )
        lock = threading.Lock()
        self._readiness = SessionReadinessCoordinator(
            lock=lock,
            sessions=lambda: mapping,
            refresh_session_meta_if_sidecar_exists=lambda _sid, **_kw: None,
            get_state=lambda _sid: state,
            log_size_or_none=log_path_size_or_none,
            confirmed_send_boundary_unresolved_for_session=(
                lambda _sid, lp, ls: consume_session_confirmed_send_boundary(session, lp, ls)
                if session is not None
                else False
            ),
            idle_from_log=self._idle_fn,
            queue_len=lambda _sid: self._queue_len_val,
            not_ready_error=RuntimeError,
        )

    def get_state(self, _session_id: str) -> dict:
        return self._state

    def _queue_len(self, _session_id: str) -> int:
        return self._queue_len_val

    def _runtime_status_from_state_and_log(self, session_id, state, log_path):
        return self._readiness.runtime_status_from_state_and_log(session_id, state, log_path)


def _snapshot(session, state, *, idle=False, queue_len=0, sessions_mapping=None, token_update=None):
    """Call the real ``message_runtime_snapshot`` with an injected fake manager."""
    manager = _SnapshotManager(
        session=session, state=state, idle=idle, queue_len=queue_len, sessions_mapping=sessions_mapping
    )
    factory = ServerRouteDepsFactory(caps=_caps(manager))
    sid = session.session_id if session is not None else "broker-1"
    return factory.message_runtime_snapshot(sid, session, token_update=token_update)


# --------------------------------------------------------------------------- #
# Listing: busy / log-idle / interrupted-idle / send-boundary resolution
# --------------------------------------------------------------------------- #


class TestListingPendingLogIdle(unittest.TestCase):
    def test_list_sessions_forces_idle_when_log_is_none(self) -> None:
        s = _session(busy=True, queue_len=0, log_path=None)
        with TemporaryDirectory() as td:
            coordinator = _list_coordinator(
                sessions={s.session_id: s},
                tmp_path=Path(td),
                probes=_probes(idle_from_log_path=lambda _sid, _path: self.fail("must not probe idle when log is None")),
            )
            out = coordinator.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0]["busy"], False)

    def test_list_sessions_uses_log_idle_over_stale_broker_busy(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"broker-1","source":"cli"}}\n', encoding="utf-8")
            s = _session(busy=True, queue_len=0, log_path=log_path)
            coordinator = _list_coordinator(
                sessions={s.session_id: s},
                tmp_path=Path(td),
                probes=_probes(idle_from_log_path=lambda _sid, _path: True),
            )
            out = coordinator.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0]["busy"], False)

    def test_list_sessions_uses_interrupted_idle_over_busy_log(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = _session(agent_backend="pi", busy=False, interrupted_idle=True, queue_len=0, log_path=log_path)
            coordinator = _list_coordinator(
                sessions={s.session_id: s},
                tmp_path=Path(td),
                probes=_probes(idle_from_log_path=lambda _sid, _path: False),
            )
            out = coordinator.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0]["busy"], False)

    def test_interrupted_idle_does_not_override_unadvanced_confirmed_send(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = _session(
                agent_backend="pi", busy=False, interrupted_idle=True, queue_len=0, log_path=log_path,
                last_send_boundary_active=True, last_send_log_path=log_path, last_send_log_size=log_path.stat().st_size,
            )
            boundary = lambda _sid, lp, ls: consume_session_confirmed_send_boundary(s, lp, ls)
            coordinator = _list_coordinator(
                sessions={s.session_id: s}, tmp_path=Path(td),
                probes=_probes(idle_from_log_path=lambda _sid, _path: False, send_boundary_unresolved=boundary),
            )
            out = coordinator.list_sessions()
            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], True)

            s.last_send_log_size -= 1
            out = coordinator.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0]["busy"], False)

    def test_stale_idle_log_does_not_override_unadvanced_confirmed_send(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            s = _session(
                agent_backend="pi", busy=False, interrupted_idle=True, queue_len=0, log_path=log_path,
                last_send_boundary_active=True, last_send_log_path=log_path, last_send_log_size=log_path.stat().st_size,
            )
            boundary = lambda _sid, lp, ls: consume_session_confirmed_send_boundary(s, lp, ls)
            coordinator = _list_coordinator(
                sessions={s.session_id: s}, tmp_path=Path(td),
                probes=_probes(idle_from_log_path=lambda _sid, _path: True, send_boundary_unresolved=boundary),
            )
            out = coordinator.list_sessions()
            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], True)

            s.last_send_log_size -= 1
            out = coordinator.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0]["busy"], False)

    def test_missing_log_does_not_override_unadvanced_confirmed_send(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "missing.jsonl"
            s = _session(
                agent_backend="pi", busy=False, interrupted_idle=True, queue_len=0, log_path=log_path,
                last_send_boundary_active=True, last_send_log_path=log_path, last_send_log_size=10,
            )
            boundary = lambda _sid, lp, ls: consume_session_confirmed_send_boundary(s, lp, ls)
            coordinator = _list_coordinator(
                sessions={s.session_id: s}, tmp_path=Path(td),
                probes=_probes(
                    idle_from_log_path=lambda _sid, _path: self.fail("missing log must not be parsed for list display"),
                    send_boundary_unresolved=boundary,
                ),
            )
            out = coordinator.list_sessions()
            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], True)

            s.last_send_log_path = Path(td) / "other.jsonl"
            out = coordinator.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0]["busy"], False)

    def test_list_sessions_skips_non_object_jsonl_rows(self) -> None:
        # The listing path delegates log parsing to the injected probe, so the
        # non-object-row robustness is owned by the real idle parser. Here we
        # assert the listing preserves idle=True reported by the probe even when
        # the log contains non-object rows (the probe stands in for the parser
        # that must tolerate them).
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('[]\n{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            s = _session(agent_backend="pi", busy=False, queue_len=0, log_path=log_path)
            coordinator = _list_coordinator(
                sessions={s.session_id: s}, tmp_path=Path(td),
                probes=_probes(idle_from_log_path=lambda _sid, _path: True),
            )
            out = coordinator.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0]["busy"], False)

    def test_no_log_confirmed_send_boundary_keeps_list_busy_until_nonempty_log(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            s = _session(
                agent_backend="pi", busy=False, interrupted_idle=True, queue_len=0, log_path=None,
                last_send_boundary_active=True, last_send_log_path=None, last_send_log_size=None,
            )
            boundary = lambda _sid, lp, ls: consume_session_confirmed_send_boundary(s, lp, ls)
            coordinator = _list_coordinator(
                sessions={s.session_id: s}, tmp_path=Path(td),
                probes=_probes(idle_from_log_path=lambda _sid, _path: True, send_boundary_unresolved=boundary),
            )
            out = coordinator.list_sessions()
            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], True)

            log_path.write_text("", encoding="utf-8")
            s.log_path = log_path
            out = coordinator.list_sessions()
            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], True)
            self.assertTrue(s.last_send_boundary_active)

            for content in ("\n", "garbage\n", "[]\n", '{"type":"message","message":{"role":"assistant","content":['):
                log_path.write_text(content, encoding="utf-8")
                out = coordinator.list_sessions()
                self.assertEqual(len(out), 1)
                self.assertIs(out[0]["busy"], True)
                self.assertTrue(s.last_send_boundary_active)

            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            out = coordinator.list_sessions()
            self.assertEqual(len(out), 1)
            self.assertIs(out[0]["busy"], False)
            self.assertFalse(s.last_send_boundary_active)
            self.assertIsNone(s.last_send_log_path)
            self.assertIsNone(s.last_send_log_size)

            s.log_path = None
            out = coordinator.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0]["busy"], False)

    def test_interrupted_idle_does_not_override_nonempty_broker_queue(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = _session(agent_backend="pi", busy=False, interrupted_idle=True, queue_len=1, log_path=log_path)
            coordinator = _list_coordinator(
                sessions={s.session_id: s}, tmp_path=Path(td),
                probes=_probes(idle_from_log_path=lambda _sid, _path: False),
            )
            out = coordinator.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0]["busy"], True)

    def test_interrupted_idle_does_not_override_busy_broker(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = _session(agent_backend="pi", busy=True, interrupted_idle=True, queue_len=0, log_path=log_path)
            coordinator = _list_coordinator(
                sessions={s.session_id: s}, tmp_path=Path(td),
                probes=_probes(idle_from_log_path=lambda _sid, _path: False),
            )
            out = coordinator.list_sessions()
        self.assertEqual(len(out), 1)
        self.assertIs(out[0]["busy"], True)

    def test_idle_from_log_path_survives_detach_after_row_snapshot(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"thread-old","source":"cli"}}\n', encoding="utf-8")
            s = _session(thread_id="thread-new", busy=True, queue_len=0, log_path=None)
            lock = threading.Lock()
            coordinator = SessionLogRuntimeCoordinator(
                lock=lock,
                sessions=lambda: {s.session_id: s},
                analyze_log_chunk=lambda _rows: (0, 0, 0, None, None, []),
                turn_context_run_settings=lambda _payload: (None, None),
                compute_idle_from_log=lambda _path: True,
                read_jsonl_from_offset=lambda _path, _off, **_kw: ([], 0),
                find_latest_token_update=lambda _path: None,
            )
            self.assertIs(coordinator.idle_from_log_path(s.session_id, log_path), True)
        self.assertEqual(s.idle_cache_log_off, -1)
        self.assertIsNone(s.idle_cache_value)


# --------------------------------------------------------------------------- #
# Message runtime snapshot: injected manager via ServerRouteDepsFactory
# --------------------------------------------------------------------------- #


class TestMessageRuntimeSnapshot(unittest.TestCase):
    def test_message_snapshot_uses_log_idle_over_stale_broker_busy(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"broker-1","source":"cli"}}\n', encoding="utf-8")
            s = _session(busy=True, queue_len=0, log_path=log_path)
            _state, busy, queue_len, _token = _snapshot(s, {"busy": True, "queue_len": 0}, idle=True)
        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_uses_interrupted_idle_over_busy_log(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = _session(agent_backend="pi", busy=True, queue_len=0, log_path=log_path)
            _state, busy, queue_len, _token = _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": True}, idle=False)
        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_rejects_interrupted_idle_before_confirmed_send_advances(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = _session(
                agent_backend="pi", busy=True, queue_len=0, log_path=log_path,
                last_send_boundary_active=True, last_send_log_path=log_path, last_send_log_size=log_path.stat().st_size,
            )
            _state, busy, queue_len, _token = _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": True}, idle=False)
            self.assertIs(busy, True)
            self.assertEqual(queue_len, 0)

            s.last_send_log_size -= 1
            _state, busy, queue_len, _token = _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": True}, idle=False)
        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_rejects_stale_idle_log_before_confirmed_send_advances(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            s = _session(
                agent_backend="pi", busy=True, queue_len=0, log_path=log_path,
                last_send_boundary_active=True, last_send_log_path=log_path, last_send_log_size=log_path.stat().st_size,
            )
            _state, busy, queue_len, _token = _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": True}, idle=True)
            self.assertIs(busy, True)
            self.assertEqual(queue_len, 0)

            s.last_send_log_size -= 1
            _state, busy, queue_len, _token = _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": True}, idle=True)
        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_rejects_missing_log_before_confirmed_send_advances(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "missing.jsonl"
            s = _session(
                agent_backend="pi", busy=True, queue_len=0, log_path=log_path,
                last_send_boundary_active=True, last_send_log_path=log_path, last_send_log_size=10,
            )
            _state, busy, queue_len, _token = _snapshot(
                s, {"busy": False, "queue_len": 0, "interrupted_idle": True},
                idle=lambda _sid: (_ for _ in ()).throw(AssertionError("missing log must not be parsed")),
            )
            self.assertIs(busy, True)
            self.assertEqual(queue_len, 0)

            s.last_send_log_path = Path(td) / "other.jsonl"
            _state, busy, queue_len, _token = _snapshot(
                s, {"busy": False, "queue_len": 0, "interrupted_idle": True},
                idle=lambda _sid: (_ for _ in ()).throw(AssertionError("missing log must not be parsed")),
            )
        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_rejects_no_log_confirmed_send_boundary_until_nonempty_log(self) -> None:
        s = _session(
            agent_backend="pi", busy=True, queue_len=0, log_path=None,
            last_send_boundary_active=True, last_send_log_path=None, last_send_log_size=None,
        )
        _state, busy, queue_len, _token = _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": True}, idle=True)
        self.assertIs(busy, True)
        self.assertEqual(queue_len, 0)
        self.assertTrue(s.last_send_boundary_active)

        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text("", encoding="utf-8")
            s.log_path = log_path
            _state, busy, queue_len, _token = _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": True}, idle=True)
            self.assertIs(busy, True)
            self.assertEqual(queue_len, 0)
            self.assertTrue(s.last_send_boundary_active)

            for content in ("\n", "garbage\n", "[]\n", '{"type":"message","message":{"role":"assistant","content":['):
                log_path.write_text(content, encoding="utf-8")
                _state, busy, queue_len, _token = _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": True}, idle=True)
                self.assertIs(busy, True)
                self.assertEqual(queue_len, 0)
                self.assertTrue(s.last_send_boundary_active)

            log_path.write_text('{"type":"message","message":{"role":"assistant","content":[],"stopReason":"aborted"}}\n', encoding="utf-8")
            _state, busy, queue_len, _token = _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": True}, idle=True)
            self.assertIs(busy, False)
            self.assertEqual(queue_len, 0)
            self.assertFalse(s.last_send_boundary_active)
            self.assertIsNone(s.last_send_log_path)
            self.assertIsNone(s.last_send_log_size)

            s.log_path = None
            _state, busy, queue_len, _token = _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": True}, idle=True)
        self.assertIs(busy, False)
        self.assertEqual(queue_len, 0)

    def test_message_snapshot_rejects_malformed_interrupted_idle(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "pi.jsonl"
            log_path.write_text('{"type":"message","message":{"role":"user","content":[{"type":"text","text":"run"}]}}\n', encoding="utf-8")
            s = _session(agent_backend="pi", log_path=log_path)
            with self.assertRaisesRegex(ValueError, "invalid broker state response"):
                _snapshot(s, {"busy": False, "queue_len": 0, "interrupted_idle": "true"}, idle=False)

    def test_message_snapshot_rejects_malformed_mocked_broker_state(self) -> None:
        s = _session(log_path=None)
        with self.assertRaisesRegex(ValueError, "invalid broker state response"):
            _snapshot(
                s, {"busy": "false", "queue_len": 0},
                idle=lambda _sid: (_ for _ in ()).throw(AssertionError("must not parse idle before broker state validates")),
            )

    def test_message_snapshot_prefers_log_token_over_stale_broker_token(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"broker-1","source":"cli"}}\n', encoding="utf-8")
            s = _session(busy=False, queue_len=0, log_path=log_path, token={"tokens_in_context": 185136})
            _state, _busy, _queue_len, token = _snapshot(s, {"busy": False, "queue_len": 0, "token": {"tokens_in_context": 0}}, idle=True)
        self.assertEqual(token, {"tokens_in_context": 185136})

    def test_message_snapshot_ignores_stale_broker_token_when_log_has_no_token_yet(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"broker-1","source":"cli"}}\n', encoding="utf-8")
            s = _session(busy=False, queue_len=0, log_path=log_path, token=None)
            _state, _busy, _queue_len, token = _snapshot(s, {"busy": False, "queue_len": 0, "token": {"tokens_in_context": 0}}, idle=True)
        self.assertIsNone(token)


# --------------------------------------------------------------------------- #
# SessionControlCoordinator.get_state
# --------------------------------------------------------------------------- #


class TestGetState(unittest.TestCase):
    def _control_coordinator(self, *, session, sock_response, tmp_path):
        lock = threading.Lock()
        sessions = {session.session_id: session}
        return SessionControlCoordinator(
            lock=lock,
            sessions=lambda: sessions,
            sock_call=lambda _sock, _req, **_kw: sock_response,
            pid_alive=lambda _pid: True,
            unlink_quiet=lambda _p: None,
            clear_deleted_session_state=lambda _sid: None,
            broker_busy_queue=broker_busy_queue,
            broker_interrupted_idle=broker_interrupted_idle,
            control_socket_call_error=OSError,
            commit_unknown_error=RuntimeError,
        )

    def test_get_state_does_not_overwrite_log_token_with_stale_broker_token(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"broker-1","source":"cli"}}\n', encoding="utf-8")
            s = _session(busy=False, queue_len=0, log_path=log_path, token={"tokens_in_context": 185136})
            coordinator = self._control_coordinator(
                session=s,
                sock_response={"busy": False, "queue_len": 0, "token": {"tokens_in_context": 0}},
                tmp_path=Path(td),
            )
            state = coordinator.get_state(s.session_id)
        self.assertEqual(state["token"], {"tokens_in_context": 0})
        self.assertEqual(s.token, {"tokens_in_context": 185136})

    def test_get_state_refreshes_interrupted_idle_cache(self) -> None:
        s = _session(agent_backend="pi", busy=False, interrupted_idle=True, queue_len=0, log_path=None)
        with TemporaryDirectory() as td:
            coordinator = self._control_coordinator(
                session=s,
                sock_response={"busy": False, "queue_len": 0, "interrupted_idle": False},
                tmp_path=Path(td),
            )
            state = coordinator.get_state(s.session_id)
        self.assertIs(state["interrupted_idle"], False)
        self.assertFalse(s.interrupted_idle)


# --------------------------------------------------------------------------- #
# SessionPruneCoordinator.refresh_session_state
# --------------------------------------------------------------------------- #


class TestRefreshSessionState(unittest.TestCase):
    def _prune_coordinator(self, *, session, sock_call, tmp_path):
        lock = threading.Lock()
        sessions = {session.session_id: session}
        return SessionPruneCoordinator(
            lock=lock,
            sessions=lambda: sessions,
            sock_call=sock_call,
            broker_busy_queue_from_state=broker_busy_queue,
            broker_interrupted_idle_from_state=broker_interrupted_idle,
            sock_error_definitely_stale=lambda _exc: False,
            pid_alive=lambda _pid: True,
            latest_launch_attempt=lambda _sid: None,
            submitted_user_messages=lambda _record: [],
            launch_failure_tail=lambda _record: "",
            which_tmux=lambda _name: None,
            tmux_pane_snapshot=lambda *a, **k: {},
            clean_optional_text=lambda value: value if isinstance(value, str) else None,
            record_launch_attempt=lambda _record: None,
            clear_deleted_session_state=lambda _sid: None,
            unlink_quiet=lambda _p: None,
        )

    def test_refresh_session_state_does_not_overwrite_log_token_with_stale_broker_token(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text('{"type":"session_meta","payload":{"id":"broker-1","source":"cli"}}\n', encoding="utf-8")
            s = _session(busy=False, queue_len=0, log_path=log_path, token={"tokens_in_context": 185136})
            coordinator = self._prune_coordinator(
                session=s,
                sock_call=lambda _sock, _req, **_kw: {"busy": False, "queue_len": 0, "token": {"tokens_in_context": 0}},
                tmp_path=Path(td),
            )
            ok, err = coordinator.refresh_session_state(s.session_id, s.sock_path)
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertEqual(s.token, {"tokens_in_context": 185136})

    def test_refresh_session_state_rejects_malformed_broker_state_without_coercion(self) -> None:
        s = _session(busy=False, queue_len=0, log_path=None)
        malformed = [
            {"busy": "false", "queue_len": 0},
            {"busy": False, "queue_len": "0"},
            {"busy": False, "queue_len": -1},
            {"busy": False, "queue_len": True},
        ]
        with TemporaryDirectory() as td:
            for state in malformed:
                with self.subTest(state=state):
                    coordinator = self._prune_coordinator(
                        session=s,
                        sock_call=lambda _sock, _req, state=state, **_kw: state,
                        tmp_path=Path(td),
                    )
                    ok, err = coordinator.refresh_session_state(s.session_id, s.sock_path)
                    self.assertFalse(ok)
                    self.assertIsInstance(err, ValueError)
                    self.assertFalse(s.busy)
                    self.assertEqual(s.queue_len, 0)


# --------------------------------------------------------------------------- #
# SessionRefreshCoordinator.refresh_session_meta
# --------------------------------------------------------------------------- #


class TestRefreshSessionMeta(unittest.TestCase):
    def test_refresh_session_meta_clears_interrupted_idle_on_log_change(self) -> None:
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
            s = _session(
                agent_backend="pi", log_path=old_log, sock_path=sock,
                interrupted_idle=True, thread_id="broker-1",
            )
            lock = threading.Lock()
            coordinator = SessionRefreshCoordinator(
                lock=lock,
                sessions=lambda: {s.session_id: s},
                prune_stale_socket_without_metadata=lambda _sid, _sock: None,
                log_invalid_sidecar_metadata=lambda _ctx, _sock, _exc: None,
                session_transport=lambda **_kw: (None, None, None),
                sock_call=lambda _sock, _req, **_kw: {},
                broker_tail_has_session_detach_marker=lambda _backend, _tail: False,
                pid_alive=lambda _pid: True,
                proc_find_open_rollout_log=lambda **_kw: None,
                proc_root=root,
                read_session_meta_or_none=lambda _path, **_kw: None,
                coerce_main_thread_log=lambda **_kw: (_kw["thread_id"], _kw["log_path"]),
                clean_optional_text=_sidecar_clean_optional_text,
                session_run_settings=lambda **_kw: (None, None, None, None),
                normalize_requested_service_tier=lambda _value: None,
                reset_log_caches=lambda session, log_off: reset_session_log_caches(session, meta_log_off=log_off),
                queue_len=lambda _sid: 0,
                maybe_drain_session_queue=lambda _sid: None,
            )
            coordinator.refresh_session_meta(s.session_id, drain_queue=False)
        self.assertEqual(s.log_path, new_log)
        self.assertFalse(s.interrupted_idle)


if __name__ == "__main__":
    unittest.main()
