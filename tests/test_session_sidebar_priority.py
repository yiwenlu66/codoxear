"""Direct coordinator/dependency-injection tests for sidebar priority / session
ordering behaviour.

Previously these tests built a ``SessionManager`` via ``__new__`` and stubbed
~13 internal methods (``_discover_existing_if_stale``, ``_prune_dead_sessions``,
``_update_meta_counters``, every ``_save_*`` saver, ``idle_from_log``,
``idle_from_log_path``, ``_maybe_drain_session_queue``, ``_sock_call``) plus
patched ``codoxear.server`` module globals (``SIDEBAR_PRIORITY_BUCKET_SECONDS``,
``time.time``, ``_current_git_branch``, ``_read_run_settings_from_log``,
``_last_conversation_ts_from_tail``, ``_process_group_alive``, ``_pid_alive``,
``_terminate_process_group``, ``_terminate_process``, ``_unlink_quiet``) and
``pathlib.Path.exists``.

They now exercise the true seams directly:

* sidebar priority / ordering / snooze / dependency clearing / queue-non-drain
  / model+reasoning+tmux projection -> ``SessionListCoordinator.list_sessions``
  built with a real ``SessionStore`` and a ``ListingRuntimeProbes`` wired to the
  real server readers. Time (``now``), the priority bucket/half-life and every
  runtime probe are injected fields, so no module-global patch is required.
* git-branch / run-settings reads happening outside the manager lock -> the
  injected ``current_git_branch`` / ``read_run_settings_from_log`` probes assert
  the shared lock is not held when they run (the listing coordinator calls them
  outside ``with lock:``), replacing the ``codoxear.server.*`` patches.
* session deletion (terminal-owned kill + dependent state clearing) ->
  ``SessionLifecycleCoordinator.delete_session`` wired to a real
  ``SessionCleanupCoordinator.clear_deleted_session_state`` and an injected
  ``sock_call`` recorder.
* socket-dead kill fallback -> ``SessionLifecycleCoordinator.kill_session`` with
  an injected ``sock_call`` that raises ``OSError`` and a recorder fallback.
* pid-only teardown without signals -> ``SessionLifecycleCoordinator.
  kill_session_via_pids`` with the OS/process/socket boundaries
  (``process_group_alive`` / ``pid_alive`` / ``terminate_process_group`` /
  ``terminate_process`` / ``unlink_quiet``) injected as fields.
* atomic edit rejection -> ``SessionUiStateCoordinator.edit_session`` with the
  real store-backed aliases / sidebar_meta.
* recent-cwd projection -> ``SessionRecentCwdCoordinator.list_recent`` reading
  the same store the listing coordinator noted cwds into.

The only retained ``patch`` targets a genuine time boundary
(``codoxear.server.time.time`` is no longer patched; ``now`` is injected). No
file under ``codoxear/`` is modified. No ``try/except`` swallows.
"""

import dataclasses
import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear import server
from codoxear.session_cleanup import SessionCleanupCoordinator
from codoxear.session_lifecycle import SessionLifecycleCoordinator
from codoxear.session_list import SessionListCoordinator
from codoxear.session_model import Session
from codoxear.session_recent_cwd import SessionRecentCwdCoordinator
from codoxear.session_runtime import ListingRuntimeProbes
from codoxear.session_runtime import log_path_size_or_none
from codoxear.session_store import SessionStore
from codoxear.session_store import SessionStorePaths
from codoxear.session_ui_state import SessionUiStateCoordinator

HALF_LIFE = server.SIDEBAR_PRIORITY_HALF_LIFE_SECONDS
BUCKET = server.SIDEBAR_PRIORITY_BUCKET_SECONDS


# --------------------------------------------------------------------------- #
# Injected-dependency builders.
# --------------------------------------------------------------------------- #


def _store(root: Path) -> SessionStore:
    """Real ``SessionStore`` rooted at a temp dir. Tests mutate its in-memory
    dicts directly; the listing/lifecycle/ui-state coordinators read and mutate
    the same store, so persistence savers can stay inert no-ops."""
    return SessionStore(
        paths=SessionStorePaths(
            aliases=root / "aliases.json",
            sidebar_meta=root / "sidebar.json",
            hidden_sessions=root / "hidden.json",
            files=root / "files.json",
            queues=root / "queues.json",
            pending_attachments=root / "pending.json",
            commit_unknown_sends=root / "unknown.json",
            recent_cwds=root / "recent.json",
            unattended=root / "unattended.json",
        ),
        file_history_max=5,
        recent_cwd_max=5,
        unattended_default_idle_minutes=server.UNATTENDED_DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=server.UNATTENDED_DEFAULT_MAX_INJECTIONS,
        clean_alias=server._clean_alias,
        clean_priority_offset=server._clean_priority_offset,
        clean_snooze_until=server._clean_snooze_until,
        clean_dependency_session_id=server._clean_dependency_session_id,
        clean_recent_cwd=server._clean_recent_cwd,
        clean_commit_unknown_send_record=lambda value: value if isinstance(value, dict) else None,
    )


def _probes(**overrides) -> ListingRuntimeProbes:
    """``ListingRuntimeProbes`` wired to the real server readers by default.
    Tests override individual probes (for example to assert the lock is not
    held, or to count history scans) instead of patching module globals."""
    probes = ListingRuntimeProbes(
        last_conversation_ts_from_tail=server._last_conversation_ts_from_tail,
        read_run_settings_from_log=lambda path, agent_backend: server._read_run_settings_from_log(
            path, agent_backend=agent_backend
        ),
        log_size_or_none=log_path_size_or_none,
        send_boundary_unresolved=lambda _sid, _path, _size: False,
        idle_from_log_path=lambda _sid, _path: True,
        current_git_branch=server._current_git_branch,
    )
    if overrides:
        return dataclasses.replace(probes, **overrides)
    return probes


def _list_coordinator(
    *,
    store: SessionStore,
    sessions: dict[str, Session],
    lock: threading.Lock,
    probes: ListingRuntimeProbes | None = None,
    now=time.time,
    priority_half_life_seconds: float = HALF_LIFE,
    priority_bucket_seconds: float = BUCKET,
    save_files=None,
    save_sidebar_meta=None,
    save_recent_cwds=None,
    include_launch_attempts: bool = False,
    resolve_session_cwd=server._resolve_session_cwd,
) -> SessionListCoordinator:
    """``SessionListCoordinator`` wired to the real store and real server
    cleaners/providers. Queues, unattended, aliases, hidden sessions and
    commit-unknown sends default to the store's own dicts so tests mutate the
    store and the coordinator observes the same state."""
    return SessionListCoordinator(
        lock=lock,
        sessions=lambda: sessions,
        queues=lambda: store.queues,
        unattended=lambda: store.unattended,
        aliases=lambda: store.aliases,
        hidden_sessions=lambda: store.hidden_sessions,
        commit_unknown_sends=lambda: store.commit_unknown_sends,
        store=store,
        discover_existing_if_stale=lambda: None,
        prune_dead_sessions=lambda: None,
        update_meta_counters=lambda: None,
        save_files=save_files or (lambda: None),
        save_sidebar_meta=save_sidebar_meta or (lambda: None),
        save_recent_cwds=save_recent_cwds or (lambda: None),
        now=now,
        runtime_probes=probes or _probes(),
        include_launch_attempts=lambda: include_launch_attempts,
        read_launch_attempts=lambda: [],
        launch_attempt_row=lambda _record: None,
        clean_unattended_cooldown_minutes=server._clean_unattended_cooldown_minutes,
        clean_unattended_remaining_injections=server._clean_unattended_remaining_injections,
        provider_choice_for_settings=server._provider_choice_for_settings,
        resolve_session_cwd=resolve_session_cwd,
        unattended_default_idle_minutes=server.UNATTENDED_DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=server.UNATTENDED_DEFAULT_MAX_INJECTIONS,
        priority_half_life_seconds=priority_half_life_seconds,
        priority_bucket_seconds=priority_bucket_seconds,
    )


def _cleanup_coordinator(
    *,
    store: SessionStore,
    sessions: dict[str, Session],
    lock: threading.Lock,
    input_locks: dict[str, object] | None = None,
    unlink_quiet=None,
) -> SessionCleanupCoordinator:
    return SessionCleanupCoordinator(
        lock=lock,
        sessions=lambda: sessions,
        store=lambda: store,
        input_locks=lambda: input_locks if input_locks is not None else {},
        unlink_quiet=unlink_quiet or (lambda _path: None),
        save_pending_attachments=lambda: None,
        save_commit_unknown_sends=lambda: None,
        save_aliases=lambda: None,
        save_sidebar_meta=lambda: None,
        save_hidden_sessions=lambda: None,
        save_unattended=lambda: None,
        save_files=lambda: None,
        save_queues=lambda: None,
    )


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
    def test_list_sessions_tolerates_malformed_cwd_metadata(self) -> None:
        # ``_resolve_session_cwd`` raises ValueError on the null-byte cwd; the
        # listing snapshot catches that, leaves ``_cwd_path_obj`` None, so the
        # git-branch probe is never invoked and ``git_branch`` is None. No
        # manager object or module patch is involved.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            s = _session(sid="target", start_ts=time.time())
            s.cwd = "/tmp/bad\x00cwd"
            sessions = {s.session_id: s}
            coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock)

            [item] = coordinator.list_sessions()

            self.assertEqual(item["session_id"], "target")
            self.assertIsNone(item["git_branch"])

    def test_list_sessions_masks_stale_unattended_enabled_zero_remaining(self) -> None:
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            s = _session(sid="target", start_ts=time.time())
            sessions = {s.session_id: s}
            store.unattended = {
                "target": {"enabled": True, "request": "A", "cooldown_minutes": 5, "remaining_injections": 0}
            }
            coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock)

            [item] = coordinator.list_sessions()

            self.assertFalse(item["unattended_enabled"])
            self.assertEqual(item["unattended_remaining_injections"], 0)

    def test_list_sessions_sorts_by_final_priority_then_recency(self) -> None:
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            now = time.time()
            recent = _session(sid="recent", start_ts=now - 100, last_chat_ts=now - 300)
            older = _session(sid="older", start_ts=now - 200, last_chat_ts=now - 16 * 3600)
            sessions = {recent.session_id: recent, older.session_id: older}
            store.sidebar_meta = {
                "recent": {"priority_offset": -0.8},
                "older": {"priority_offset": 0.2},
            }
            coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock, now=lambda: now)

            rows = coordinator.list_sessions()

            self.assertEqual([row["session_id"] for row in rows], ["older", "recent"])
            self.assertAlmostEqual(rows[0]["final_priority"], 0.45, delta=0.04)
            self.assertAlmostEqual(rows[1]["final_priority"], 0.19, delta=0.04)

    def test_list_sessions_priority_payload_is_bucketed_for_etag_stability(self) -> None:
        # The clock and bucket size are injected coordinator fields, so the
        # ``codoxear.server.time.time`` and ``SIDEBAR_PRIORITY_BUCKET_SECONDS``
        # module-global patches are gone. The session carries a fixed
        # last_chat_ts so elapsed time is driven entirely by the injected clock.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            s = _session(sid="bucket", start_ts=1000.0, last_chat_ts=1000.0)
            sessions = {s.session_id: s}
            clock = {"t": 1005.0}
            coordinator = _list_coordinator(
                store=store,
                sessions=sessions,
                lock=lock,
                now=lambda: clock["t"],
                priority_bucket_seconds=10.0,
            )

            first = coordinator.list_sessions()[0]
            clock["t"] = 1009.0
            same_bucket = coordinator.list_sessions()[0]
            clock["t"] = 1011.0
            next_bucket = coordinator.list_sessions()[0]

            self.assertEqual(first["time_priority"], same_bucket["time_priority"])
            self.assertEqual(first["base_priority"], same_bucket["base_priority"])
            self.assertLess(next_bucket["time_priority"], same_bucket["time_priority"])

    def test_list_sessions_does_not_drain_queue(self) -> None:
        # ``SessionListCoordinator.list_sessions`` never invokes a queue-drain
        # step: it only reads queue length into the row. The structural guarantee
        # is expressed by the coordinator having no drain hook; the test asserts
        # both the projected queue_len and that the underlying store queue is
        # left untouched.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            now = time.time()
            current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 50)
            sessions = {current.session_id: current}
            store.queues = {current.session_id: [{"id": "q1", "text": "queued", "created_ts": now}]}
            coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock)

            rows = coordinator.list_sessions()

            self.assertEqual(rows[0]["queue_len"], 1)
            self.assertEqual(store.queues[current.session_id], [{"id": "q1", "text": "queued", "created_ts": now}])

    def test_list_sessions_clears_expired_snooze_and_stale_dependency(self) -> None:
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            now = time.time()
            current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 50)
            sessions = {current.session_id: current}
            store.sidebar_meta = {
                "current": {
                    "priority_offset": 0.1,
                    "snooze_until": now - 5,
                    "dependency_session_id": "missing",
                }
            }
            coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock, now=lambda: now)

            rows = coordinator.list_sessions()

            self.assertEqual(rows[0]["session_id"], "current")
            self.assertFalse(rows[0]["snoozed"])
            self.assertFalse(rows[0]["blocked"])
            self.assertIsNone(rows[0]["snooze_until"])
            self.assertIsNone(rows[0]["dependency_session_id"])

    def test_list_sessions_reads_git_branch_outside_manager_lock(self) -> None:
        # ``build_runtime_enriched_session_rows`` calls the git-branch probe
        # outside ``with lock:``, so the injected probe can assert the lock is
        # not held. This replaces the ``codoxear.server._current_git_branch``
        # module-global patch.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            now = time.time()
            current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 50)
            sessions = {current.session_id: current}

            def branch_lookup(_cwd: Path) -> str:
                self.assertFalse(lock.locked())
                return "feature/outside-lock"

            coordinator = _list_coordinator(
                store=store,
                sessions=sessions,
                lock=lock,
                probes=_probes(current_git_branch=branch_lookup),
            )
            rows = coordinator.list_sessions()

            self.assertEqual(rows[0]["git_branch"], "feature/outside-lock")

    def test_list_sessions_reads_log_run_settings_outside_manager_lock(self) -> None:
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            now = time.time()
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text(
                '{"type":"session_meta","payload":{"id":"current","source":"cli"}}\n', encoding="utf-8"
            )
            current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 50)
            current.log_path = log_path
            current.model_provider = None
            current.model = None
            current.reasoning_effort = None
            sessions = {current.session_id: current}

            def read_settings(_path: Path, agent_backend: str = "codex") -> tuple[str, str, str]:
                self.assertFalse(lock.locked())
                self.assertEqual(agent_backend, "codex")
                return "openai", "gpt-5.4", "high"

            coordinator = _list_coordinator(
                store=store,
                sessions=sessions,
                lock=lock,
                probes=_probes(
                    read_run_settings_from_log=read_settings,
                ),
            )
            rows = coordinator.list_sessions()

            self.assertEqual(rows[0]["model_provider"], "openai")
            self.assertEqual(rows[0]["model"], "gpt-5.4")
            self.assertEqual(rows[0]["reasoning_effort"], "high")
            self.assertEqual(current.model, "gpt-5.4")

    def test_delete_session_kills_terminal_owned_and_clears_dependents(self) -> None:
        # ``SessionLifecycleCoordinator.delete_session`` is wired with an
        # injected ``sock_call`` recorder (socket boundary) and a real
        # ``SessionCleanupCoordinator.clear_deleted_session_state`` backed by
        # the shared store, so dependent sidebar metadata, queues, unattended
        # and per-cwd file history are cleared through the true code path.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            now = time.time()
            blocked = _session(sid="blocked", start_ts=now - 100, last_chat_ts=now - 10)
            target = _session(sid="target", start_ts=now - 200, last_chat_ts=now - 20, owned=False)
            sessions = {blocked.session_id: blocked, target.session_id: target}
            store.sidebar_meta = {
                "blocked": {"priority_offset": 0.0, "dependency_session_id": "target"},
                "target": {"priority_offset": 0.5},
            }
            store.queues = {"target": ["queued"]}
            store.unattended = {"target": {"enabled": True, "request": "x"}}
            store.files = {"cwd:/tmp/target": ["/tmp/target/a.py"]}
            called = {"shutdown": 0}

            def sock_call(_sock, _payload, *, timeout_s: float = 1.0) -> dict:
                called["shutdown"] += 1
                return {"ok": True}

            cleanup = _cleanup_coordinator(store=store, sessions=sessions, lock=lock)
            hide_calls: list[str] = []
            coordinator = SessionLifecycleCoordinator(
                lock=lock,
                sessions=lambda: sessions,
                sock_call=sock_call,
                process_group_alive=lambda _pgid: False,
                pid_alive=lambda _pid: False,
                terminate_process_group=lambda *a, **kw: True,
                terminate_process=lambda *a, **kw: True,
                unlink_quiet=lambda _p: None,
                commit_unknown_sends=lambda: store.commit_unknown_sends,
                queue_has_recovery_items_locked=lambda _sid: False,
                clear_deleted_session_state=cleanup.clear_deleted_session_state,
                read_launch_attempts=lambda: [],
                launch_attempt_row=lambda _r: None,
                hide_session=lambda sid: hide_calls.append(sid),
            )

            ok = coordinator.delete_session("target")

            self.assertTrue(ok)
            self.assertEqual(called["shutdown"], 1)
            self.assertNotIn("target", store.sidebar_meta)
            self.assertNotIn("target", store.queues)
            self.assertNotIn("target", store.unattended)
            self.assertNotIn("cwd:/tmp/target", store.files)
            self.assertIsNone(store.sidebar_meta["blocked"].get("dependency_session_id"))

    def test_kill_session_falls_back_to_pid_teardown_when_socket_is_dead(self) -> None:
        # ``sock_call`` raising OSError is the socket-boundary condition; the
        # coordinator routes to ``kill_session_via_pids_fallback`` when the
        # socket is dead. The fallback is injected as a recorder so the test
        # asserts the routing contract without patching the pid-teardown method.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            s = _session(sid="target", start_ts=time.time() - 10, last_chat_ts=None, owned=False)
            sessions = {s.session_id: s}
            fallback_calls: list[Session] = []

            def sock_call(_sock, _payload, *, timeout_s: float = 1.0) -> dict:
                raise OSError("dead socket")

            coordinator = SessionLifecycleCoordinator(
                lock=lock,
                sessions=lambda: sessions,
                sock_call=sock_call,
                process_group_alive=lambda _pgid: False,
                pid_alive=lambda _pid: False,
                terminate_process_group=lambda *a, **kw: True,
                terminate_process=lambda *a, **kw: True,
                unlink_quiet=lambda _p: None,
                commit_unknown_sends=lambda: store.commit_unknown_sends,
                queue_has_recovery_items_locked=lambda _sid: False,
                clear_deleted_session_state=lambda *a, **kw: None,
                read_launch_attempts=lambda: [],
                launch_attempt_row=lambda _r: None,
                hide_session=lambda _sid: None,
                kill_session_via_pids_fallback=lambda session: (fallback_calls.append(session), True)[1],
            )

            ok = coordinator.kill_session("target")

            self.assertTrue(ok)
            self.assertEqual(fallback_calls, [s])

    def test_kill_session_via_pids_prunes_stale_metadata_without_signals(self) -> None:
        # OS/process/socket boundaries (process-group liveness, pid liveness,
        # signal sending, socket+sidecar unlink) are injected as coordinator
        # fields rather than patched module globals. With both liveness probes
        # False the coordinator unlinks the stale socket + sidecar and skips
        # signal sending entirely.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            s = _session(sid="target", start_ts=time.time() - 10, last_chat_ts=None, owned=False)
            sessions = {s.session_id: s}
            kill_group_calls: list[list[int]] = []
            kill_proc_calls: list[int] = []
            unlink_paths: list[Path] = []

            def terminate_process_group(pid: int, *, wait_seconds: float = 1.0) -> bool:
                kill_group_calls.append([pid])
                return True

            def terminate_process(pid: int, *, wait_seconds: float = 1.0) -> bool:
                kill_proc_calls.append(pid)
                return True

            def unlink_quiet(path: Path) -> None:
                unlink_paths.append(path)

            coordinator = SessionLifecycleCoordinator(
                lock=lock,
                sessions=lambda: sessions,
                sock_call=lambda *a, **kw: {"ok": True},
                process_group_alive=lambda _pgid: False,
                pid_alive=lambda _pid: False,
                terminate_process_group=terminate_process_group,
                terminate_process=terminate_process,
                unlink_quiet=unlink_quiet,
                commit_unknown_sends=lambda: store.commit_unknown_sends,
                queue_has_recovery_items_locked=lambda _sid: False,
                clear_deleted_session_state=lambda *a, **kw: None,
                read_launch_attempts=lambda: [],
                launch_attempt_row=lambda _r: None,
                hide_session=lambda _sid: None,
            )

            ok = coordinator.kill_session_via_pids(s)

            self.assertTrue(ok)
            self.assertEqual(kill_group_calls, [])
            self.assertEqual(kill_proc_calls, [])
            self.assertEqual(unlink_paths, [s.sock_path, s.sock_path.with_suffix(".json")])

    def test_edit_session_is_atomic_when_dependency_invalid(self) -> None:
        # ``SessionUiStateCoordinator.edit_session`` validates the dependency
        # before mutating aliases or sidebar_meta; the real store-backed maps
        # express the atomicity (nothing changes when validation raises).
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            now = time.time()
            s = _session(sid="edit", start_ts=now - 100, last_chat_ts=now - 20)
            sessions = {s.session_id: s}
            store.aliases = {"edit": "old name"}
            store.sidebar_meta = {"edit": {"priority_offset": 0.1}}
            coordinator = SessionUiStateCoordinator(
                lock=lock,
                sessions=lambda: sessions,
                aliases=lambda: store.aliases,
                set_aliases=lambda _value: None,
                sidebar_meta=lambda: store.sidebar_meta,
                set_sidebar_meta=lambda _value: None,
                hidden_sessions=lambda: store.hidden_sessions,
                set_hidden_sessions=lambda _value: None,
                save_aliases=lambda: None,
                save_sidebar_meta=lambda: None,
                save_hidden_sessions=lambda: None,
                clean_alias=server._clean_alias,
                clean_priority_offset=server._clean_priority_offset,
                clean_snooze_until=server._clean_snooze_until,
                clean_dependency_session_id=server._clean_dependency_session_id,
            )

            with self.assertRaisesRegex(ValueError, "dependency session not found"):
                coordinator.edit_session(
                    "edit",
                    name="new name",
                    priority_offset=0.2,
                    snooze_until=None,
                    dependency_session_id="missing",
                )

            self.assertEqual(store.aliases["edit"], "old name")
            self.assertEqual(store.sidebar_meta["edit"]["priority_offset"], 0.1)

    def test_list_sessions_uses_start_ts_when_log_has_no_sidebar_relevant_message(self) -> None:
        # The history-scan probe is injected (asserting the lock is not held
        # while it runs) and returns None, so ``updated_ts`` falls back to
        # ``start_ts``. The log path is a real temp file, replacing the
        # ``pathlib.Path.exists`` module-global patch.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            log_path = Path(td) / "rollout.jsonl"
            log_path.write_text(
                '{"type":"session_meta","payload":{"id":"nologmsg","source":"cli"}}\n', encoding="utf-8"
            )
            s = _session(sid="nologmsg", start_ts=123.0, last_chat_ts=None)
            s.log_path = log_path
            sessions = {s.session_id: s}

            def no_conversation_ts(_path: Path) -> None:
                self.assertFalse(lock.locked())
                return None

            coordinator = _list_coordinator(
                store=store,
                sessions=sessions,
                lock=lock,
                probes=_probes(last_conversation_ts_from_tail=no_conversation_ts),
            )
            rows = coordinator.list_sessions()

            self.assertEqual(rows[0]["updated_ts"], 123.0)

    def test_list_sessions_backfills_updated_ts_from_large_preexisting_log(self) -> None:
        # Real ``_last_conversation_ts_from_tail`` probe + a real log file: the
        # history backfill extracts the user-message timestamp. No patches.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            log_path = Path(td) / "rollout-2026-03-17T00-00-00-eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee.jsonl"
            now = time.time()
            user_ts = now - 30
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "session_meta", "payload": {"id": "current", "source": "cli"}}),
                        json.dumps(
                            {
                                "type": "event_msg",
                                "payload": {"type": "user_message", "message": "real turn"},
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
                        json.dumps({"type": "response_item", "payload": {"type": "reasoning"}, "ts": now - 10}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            current = _session(sid="current", start_ts=10.0, last_chat_ts=None)
            current.log_path = log_path
            sessions = {current.session_id: current}
            coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock)

            rows = coordinator.list_sessions()

            self.assertAlmostEqual(rows[0]["updated_ts"], user_ts, places=3)

    def test_list_sessions_scans_preexisting_history_only_once(self) -> None:
        # The injected history-scan probe counts calls. After the first scan
        # ``last_chat_history_scanned`` flips True on the session, so the second
        # listing does not re-scan. This replaces the
        # ``codoxear.server._last_conversation_ts_from_tail`` module-global patch.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
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
            sessions = {current.session_id: current}

            scan_calls: list[Path] = []

            def counting_tail(path: Path) -> None:
                scan_calls.append(path)
                return None

            coordinator = _list_coordinator(
                store=store,
                sessions=sessions,
                lock=lock,
                probes=_probes(last_conversation_ts_from_tail=counting_tail),
            )
            rows1 = coordinator.list_sessions()
            rows2 = coordinator.list_sessions()

            self.assertEqual(len(scan_calls), 1)
            self.assertEqual(rows1[0]["updated_ts"], 123.0)
            self.assertEqual(rows2[0]["updated_ts"], 123.0)

    def test_recent_cwds_include_backfilled_history_and_live_sessions(self) -> None:
        # The listing coordinator notes the live session cwd into the store;
        # ``SessionRecentCwdCoordinator.list_recent`` reads the same store, so
        # the live cwd and a pre-existing backfilled cwd both appear, ordered by
        # recency.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            store.recent_cwds = {"/repo/ended": 100.0}
            now = time.time()
            current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 5)
            sessions = {current.session_id: current}
            list_coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock)
            recent_coordinator = SessionRecentCwdCoordinator(
                lock=lock,
                store=lambda: store,
                iter_session_logs=lambda: [],
                resume_candidate_from_log=lambda _path: None,
                save_recent_cwds=lambda: None,
                now=time.time,
            )

            list_coordinator.list_sessions()

            self.assertEqual(recent_coordinator.list_recent(limit=4), ["/tmp/current", "/repo/ended"])

    def test_list_sessions_persists_new_recent_cwd_once(self) -> None:
        # The injected ``save_recent_cwds`` recorder captures persistence. The
        # first listing notes a new cwd (dirty -> save); the second listing
        # finds the same cwd at the same timestamp (not dirty -> no save).
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            save_calls: list[bool] = []
            now = time.time()
            current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 5)
            sessions = {current.session_id: current}
            recent_recorder = lambda: save_calls.append(True)  # noqa: E731
            coordinator = _list_coordinator(
                store=store,
                sessions=sessions,
                lock=lock,
                save_recent_cwds=recent_recorder,
            )

            coordinator.list_sessions()
            coordinator.list_sessions()

            self.assertEqual(save_calls, [True])
            recent_coordinator = SessionRecentCwdCoordinator(
                lock=lock,
                store=lambda: store,
                iter_session_logs=lambda: [],
                resume_candidate_from_log=lambda _path: None,
                save_recent_cwds=lambda: None,
                now=time.time,
            )
            self.assertEqual(recent_coordinator.list_recent(limit=1), ["/tmp/current"])

    def test_list_sessions_exposes_model_and_reasoning_effort(self) -> None:
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            now = time.time()
            current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 5)
            current.model = "gpt-5.4"
            current.reasoning_effort = "xhigh"
            sessions = {current.session_id: current}
            coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock)

            rows = coordinator.list_sessions()

            self.assertEqual(rows[0]["model"], "gpt-5.4")
            self.assertEqual(rows[0]["reasoning_effort"], "xhigh")

    def test_list_sessions_exposes_tmux_transport(self) -> None:
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            now = time.time()
            current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 5)
            current.transport = "tmux"
            current.tmux_session = "codoxear"
            current.tmux_window = "current-abcd12"
            sessions = {current.session_id: current}
            coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock)

            rows = coordinator.list_sessions()

            self.assertEqual(rows[0]["transport"], "tmux")
            self.assertEqual(rows[0]["tmux_session"], "codoxear")
            self.assertEqual(rows[0]["tmux_window"], "current-abcd12")

    def test_list_sessions_falls_back_to_log_run_settings(self) -> None:
        # Real ``_read_run_settings_from_log`` probe + a real log file carrying
        # a ``turn_context`` payload: run-settings backfill fills model and
        # reasoning effort. No module-global patch.
        with TemporaryDirectory() as td:
            store = _store(Path(td))
            lock = threading.Lock()
            now = time.time()
            current = _session(sid="current", start_ts=now - 100, last_chat_ts=now - 5)
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
            sessions = {current.session_id: current}
            coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock)

            rows = coordinator.list_sessions()

            self.assertEqual(rows[0]["model"], "gpt-5.4")
            self.assertEqual(rows[0]["reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
