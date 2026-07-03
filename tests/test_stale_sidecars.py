"""Direct coordinator/dependency-injection tests for stale-sidecar cleanup.

Previously these tests drove the cleanup path through ``SessionManager`` method
calls (``SessionManager._discover_existing`` / ``SessionManager.refresh_session_meta``)
on a ``SessionManager.__new__`` scaffold whose ``_save_*`` / ``_sock_call`` methods
were lambda-replaced, and redirected cleanup at the live socket directory by
monkeypatching the ``codoxear.server.SOCK_DIR`` module global (11 patch sites)
plus mutating the ``codoxear.server._INVALID_SESSION_META_WARNINGS`` module global.

They now exercise the true seams directly:

* sidecar-format validation + discovery -> ``codoxear.session_discovery.discover_sessions``
  with a ``DiscoveryDeps`` whose process-liveness (``pid_alive``), /proc log scan
  (``proc_find_open_rollout_log``), broker control socket (``sock_call``), and
  session-meta log reader (``read_session_meta_or_none``) are injected at the
  OS / process / socket / filesystem boundaries; ``broker_busy_queue_from_state``
  and ``broker_interrupted_idle_from_state`` use the real
  ``codoxear.session_runtime`` parsers so malformed broker state still raises.
* discovery-result application (session upsert, stale pruning, recent-cwd recall)
  -> ``codoxear.session_discovery_registry.SessionDiscoveryRegistryCoordinator``
  built over a real ``codoxear.session_store.SessionStore`` and the real
  ``codoxear.session_cleanup.SessionCleanupCoordinator`` prune path.
* the force-discovery staleness gate and ``last_discover_ts`` projection ->
  ``codoxear.session_manager_discovery.discover_existing_for_manager`` wired to a
  thin facade that delegates ``_discovery_deps`` / ``_apply_discovery_result`` to
  the coordinators above (the only manager surface still exercised, because the
  timestamp projection lives in that wrapper).
* per-session sidecar refresh -> ``codoxear.session_refresh.SessionRefreshCoordinator``
  with the same injected boundaries plus the real ``sidecar_metadata`` /
  ``session_runtime`` helpers.
* once-per-(context,path) invalid-session-meta warning -> the real
  ``codoxear.session_log_metadata.read_session_meta_or_none`` with a locally-owned
  ``invalid_warnings`` set and injected stderr stream.

No ``codoxear.server.SOCK_DIR`` / ``_INVALID_SESSION_META_WARNINGS`` monkeypatching
remains, no ``SessionManager`` is constructed, and no ``patch`` calls remain. No
file under ``codoxear/`` is modified. No ``try/except`` swallows.
"""

import contextlib
import io
import json
import math
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear import server
from codoxear.process_runtime import pid_alive as _pid_alive_impl
from codoxear.session_cleanup import SessionCleanupCoordinator
from codoxear.session_discovery import DiscoveryDeps
from codoxear.session_discovery import discover_sessions
from codoxear.session_discovery_registry import SessionDiscoveryRegistryCoordinator
from codoxear.session_log_discovery import proc_find_open_rollout_log as _proc_find_open_rollout_log_impl
from codoxear.session_log_metadata import read_session_meta_or_none as _read_session_meta_or_none_impl
from codoxear.session_manager_discovery import discover_existing_for_manager
from codoxear.session_manager_store import create_session_store
from codoxear.session_manager_store import session_store_paths
from codoxear.session_refresh import SessionRefreshCoordinator
from codoxear.session_refresh import broker_tail_has_session_detach_marker
from codoxear.session_registry import SessionRegistry
from codoxear.session_resume import coerce_main_thread_log as _coerce_main_thread_log_impl
from codoxear.session_runtime import broker_busy_queue
from codoxear.session_runtime import broker_interrupted_idle
from codoxear.session_runtime import reset_session_log_caches
from codoxear.session_runtime import session_run_settings_from_meta
from codoxear.session_runtime import session_transport_from_meta
from codoxear.session_store import SessionStore
from codoxear.sidecar_metadata import _clean_optional_text
from codoxear.sidecar_metadata import log_invalid as _log_invalid_sidecar_metadata
from codoxear.util import is_subagent_session_meta
from codoxear.util import subagent_parent_thread_id


# --------------------------------------------------------------------------- #
# Real-implementation adapters for the OS / process / socket / filesystem
# boundaries. Each is the genuine product helper, not a fake; only the inputs
# they close over are test-controlled.
# --------------------------------------------------------------------------- #


def _unlink_quiet(path: Path) -> None:
    """Filesystem-boundary unlink (mirrors ``server._unlink_quiet``)."""
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _read_session_meta_or_none(log_path, *args, **kwargs):
    """Adapter accepting both discovery (positional) and refresh (keyword)
    calling conventions, delegating to the real server-configured reader.

    This is a real implementation reference (not a monkeypatch seam): it calls
    ``codoxear.server._read_session_meta_or_none`` unchanged, exactly as the
    product's ``SessionManager._discovery_deps`` wiring does."""
    agent_backend = kwargs.get("agent_backend")
    if agent_backend is None and args:
        agent_backend = args[0]
    context = kwargs.get("context")
    if context is None and len(args) >= 2:
        context = args[1]
    if context is None:
        context = "discovery"
    return server._read_session_meta_or_none(log_path, agent_backend=agent_backend, context=context)


def _coerce_main_thread_log(thread_id, log_path):
    """Real ``coerce_main_thread_log`` with its log-reader / subagent helpers
    injected. For these fixtures (``source: cli`` session_meta) it is identity,
    but wiring the real impl keeps subagent coercion semantics covered."""
    return _coerce_main_thread_log_impl(
        thread_id=thread_id,
        log_path=log_path,
        read_session_meta_or_none=_read_session_meta_or_none,
        is_subagent_session_meta=is_subagent_session_meta,
        subagent_parent_thread_id=subagent_parent_thread_id,
        find_session_log_for_session_id=lambda _parent: None,
    )


def _make_store(root: Path) -> SessionStore:
    """Real ``SessionStore`` rooted at ``root`` with the real cleaners."""
    return create_session_store(
        paths=session_store_paths(
            aliases=root / "session_aliases.json",
            sidebar_meta=root / "session_sidebar.json",
            hidden_sessions=root / "hidden_sessions.json",
            files=root / "session_files.json",
            queues=root / "session_queues.json",
            pending_attachments=root / "pending_attachments.json",
            commit_unknown_sends=root / "commit_unknown_sends.json",
            recent_cwds=root / "recent_cwds.json",
            unattended=root / "unattended.json",
        ),
        file_history_max=8,
        recent_cwd_max=8,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_alias=lambda raw: str(raw).strip() if isinstance(raw, str) and raw.strip() else "",
        clean_priority_offset=lambda raw: 0.0,
        clean_snooze_until=lambda raw: None,
        clean_dependency_session_id=lambda raw: None,
        clean_recent_cwd=lambda raw: str(raw).strip() if isinstance(raw, str) and raw.strip() else None,
        clean_commit_unknown_send_record=lambda raw: dict(raw) if isinstance(raw, dict) else None,
    )


def _write_valid_sidecar(sock: Path, *, root: Path, log_path: Path | None = None) -> None:
    if log_path is None:
        log_path = root / f"{sock.stem}.jsonl"
        log_path.write_text(json.dumps({"type": "session_meta", "payload": {"id": sock.stem, "source": "cli"}}) + "\n", encoding="utf-8")
    sock.with_suffix(".json").write_text(
        json.dumps(
            {
                "session_id": sock.stem,
                "agent_backend": "codex",
                "codex_pid": 0,
                "broker_pid": 0,
                "cwd": str(root),
                "log_path": str(log_path),
                "start_ts": 123.0,
            }
        ),
        encoding="utf-8",
    )


def _session(session_id: str, sock: Path) -> object:
    from codoxear.session_model import Session

    return Session(
        session_id=session_id,
        thread_id=session_id,
        broker_pid=0,
        codex_pid=0,
        agent_backend="codex",
        owned=False,
        start_ts=time.time(),
        cwd="/tmp",
        log_path=None,
        sock_path=sock,
    )


class _DiscoveryHarness:
    """Owns the mutable registry/store state discovery closes over and wires the
    real ``discover_existing_for_manager`` wrapper plus the registry / cleanup
    coordinators.

    Boundaries injected (each replaces a former ``codoxear.server`` patch or
    ``SessionManager`` lambda-rebound method):

    * ``pid_alive``                   -> process liveness (process boundary).
    * ``proc_find_open_rollout_log``  -> /proc open-file scan (process/filesystem).
    * ``sock_call``                   -> broker control socket IPC (socket/network).
    * ``read_session_meta_or_none``   -> backend session-log reader (filesystem).
    * ``now``                         -> wall clock for the staleness gate (time).
    * ``unlink_quiet``                -> stale socket/metadata removal (filesystem).
    """

    def __init__(self, *, root: Path, sock_dir: Path, hidden_sessions: set[str] | None = None) -> None:
        self.root = root
        self.sock_dir = sock_dir
        self.proc_root = root / "proc"
        # Named ``_registry`` so ``session_registry_for_manager`` resolves this
        # exact instance (and projects ``last_discover_ts`` onto it) rather than
        # lazily allocating a fresh one.
        self._registry = SessionRegistry()
        self.store = _make_store(root)
        if hidden_sessions is not None:
            self.store.hidden_sessions = set(hidden_sessions)
        self._cleanup = SessionCleanupCoordinator(
            lock=self._registry.lock,
            sessions=lambda: self._registry.sessions,
            store=lambda: self.store,
            input_locks=lambda: self._registry.input_locks,
            unlink_quiet=_unlink_quiet,
            save_pending_attachments=lambda: None,
            save_commit_unknown_sends=lambda: None,
            save_aliases=lambda: None,
            save_sidebar_meta=lambda: None,
            save_hidden_sessions=lambda: None,
            save_unattended=lambda: None,
            save_files=lambda: None,
            save_queues=lambda: None,
        )
        self._registry_coord = SessionDiscoveryRegistryCoordinator(
            lock=self._registry.lock,
            sessions=lambda: self._registry.sessions,
            pending_attachment_ids=lambda: self.store.pending_attachment_ids,
            commit_unknown_sends=lambda: self.store.commit_unknown_sends,
            reset_log_caches=lambda session, log_off: reset_session_log_caches(session, meta_log_off=log_off),
            record_launch_attempt=lambda _record: None,
            prune_stale_socket_without_metadata=self._cleanup.prune_stale_socket_without_metadata,
            unhide_session=self._unhide_session,
            unlink_quiet=_unlink_quiet,
            remember_recent_cwd=self.store.remember_recent_cwd,
            save_recent_cwds=lambda: None,
        )

    # -- manager-facade surface used by discover_existing_for_manager ----------
    @property
    def registry(self) -> SessionRegistry:
        return self._registry

    @property
    def _hidden_sessions(self) -> set[str]:
        return self.store.hidden_sessions

    def _unhide_session(self, session_id: str) -> None:
        self.store.hidden_sessions.discard(session_id)

    def _discovery_deps(self) -> DiscoveryDeps:
        return DiscoveryDeps(
            pid_alive=_pid_alive_impl,
            proc_find_open_rollout_log=_proc_find_open_rollout_log_impl,
            read_session_meta_or_none=_read_session_meta_or_none,
            coerce_main_thread_log=_coerce_main_thread_log,
            session_transport=lambda meta: session_transport_from_meta(meta=meta, clean_optional_text=_clean_optional_text),
            session_run_settings=lambda meta, log_path, agent_backend: session_run_settings_from_meta(
                meta=meta,
                log_path=log_path,
                agent_backend=agent_backend,
                clean_optional_text=_clean_optional_text,
                normalize_requested_preferred_auth_method=server._normalize_requested_preferred_auth_method,
                display_reasoning_effort=server._display_reasoning_effort,
                display_pi_reasoning_effort=server._display_pi_reasoning_effort,
                normalize_requested_cc_reasoning_effort=server._normalize_requested_cc_reasoning_effort,
                read_run_settings_from_log=server._read_run_settings_from_log,
            ),
            sock_call=lambda _sock, _req, _timeout_s: {"busy": False, "queue_len": 0, "token": None},
            broker_busy_queue_from_state=broker_busy_queue,
            broker_interrupted_idle_from_state=broker_interrupted_idle,
            sock_error_definitely_stale=lambda _exc: False,
            token_update_finder=lambda _log_path: None,
        )

    def _apply_discovery_result(self, result) -> None:
        self._registry_coord.apply_result(result)

    # -- driver ---------------------------------------------------------------
    def discover(self, *, force: bool = True, now=time.time) -> None:
        discover_existing_for_manager(
            self,
            force=force,
            discover_min_interval_seconds=0.0,
            sock_dir=self.sock_dir,
            proc_root=self.proc_root,
            discover_sessions=discover_sessions,
            now=now,
        )


def _refresh_coordinator(
    *,
    sessions: dict,
    proc_root: Path,
    cleanup: SessionCleanupCoordinator,
) -> SessionRefreshCoordinator:
    return SessionRefreshCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        prune_stale_socket_without_metadata=cleanup.prune_stale_socket_without_metadata,
        log_invalid_sidecar_metadata=_log_invalid_sidecar_metadata,
        session_transport=lambda meta: session_transport_from_meta(meta=meta, clean_optional_text=_clean_optional_text),
        sock_call=lambda _sock, _req, **_kw: {},
        broker_tail_has_session_detach_marker=broker_tail_has_session_detach_marker,
        pid_alive=_pid_alive_impl,
        proc_find_open_rollout_log=_proc_find_open_rollout_log_impl,
        proc_root=proc_root,
        read_session_meta_or_none=_read_session_meta_or_none,
        coerce_main_thread_log=_coerce_main_thread_log,
        clean_optional_text=_clean_optional_text,
        session_run_settings=lambda meta, log_path, agent_backend: session_run_settings_from_meta(
            meta=meta,
            log_path=log_path,
            agent_backend=agent_backend,
            clean_optional_text=_clean_optional_text,
            normalize_requested_preferred_auth_method=server._normalize_requested_preferred_auth_method,
            display_reasoning_effort=server._display_reasoning_effort,
            display_pi_reasoning_effort=server._display_pi_reasoning_effort,
            normalize_requested_cc_reasoning_effort=server._normalize_requested_cc_reasoning_effort,
            read_run_settings_from_log=server._read_run_settings_from_log,
        ),
        normalize_requested_service_tier=server._normalize_requested_service_tier,
        reset_log_caches=lambda session, log_off: reset_session_log_caches(session, meta_log_off=log_off),
        queue_len=lambda _sid: 0,
        maybe_drain_session_queue=lambda _sid: None,
    )


def _refresh_for_root(root: Path, *, sessions: dict) -> SessionRefreshCoordinator:
    """Build a refresh coordinator sharing a real cleanup coordinator + store
    rooted at ``root`` so prune-side-effects (unlink, state clear) are real."""
    store = _make_store(root)
    cleanup = SessionCleanupCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        store=lambda: store,
        input_locks=lambda: {},
        unlink_quiet=_unlink_quiet,
        save_pending_attachments=lambda: None,
        save_commit_unknown_sends=lambda: None,
        save_aliases=lambda: None,
        save_sidebar_meta=lambda: None,
        save_hidden_sessions=lambda: None,
        save_unattended=lambda: None,
        save_files=lambda: None,
        save_queues=lambda: None,
    )
    return _refresh_coordinator(sessions=sessions, proc_root=root / "proc", cleanup=cleanup)


class TestStaleSidecars(unittest.TestCase):
    def test_discovery_prunes_sock_without_metadata_sidecar(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            sock = sock_dir / "stale.sock"
            sock.touch()
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir, hidden_sessions={"stale"})
            h.registry.sessions["stale"] = _session("stale", sock)
            h.store.aliases["stale"] = "Stale"
            h.store.queues["stale"] = [{"id": "q", "text": "later"}]

            h.discover(force=True)

            self.assertFalse(sock.exists())
            self.assertNotIn("stale", h.registry.sessions)
            self.assertNotIn("stale", h.store.hidden_sessions)
            self.assertNotIn("stale", h.store.aliases)
            self.assertNotIn("stale", h.store.queues)
            self.assertGreater(h.registry.last_discover_ts, 0)

    def test_refresh_prunes_existing_session_when_sidecar_disappears(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock = root / "gone.sock"
            sock.touch()
            sessions: dict = {"gone": _session("gone", sock)}
            coord = _refresh_for_root(root, sessions=sessions)

            coord.refresh_session_meta("gone")

            self.assertFalse(sock.exists())
            self.assertNotIn("gone", sessions)

    def test_discovery_skips_malformed_json_sidecar_and_keeps_good_sidecar(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            bad_sock = sock_dir / "bad.sock"
            bad_sock.touch()
            bad_sock.with_suffix(".json").write_text("{not-json}\n", encoding="utf-8")
            good_sock = sock_dir / "good.sock"
            good_sock.touch()
            _write_valid_sidecar(good_sock, root=root)
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                h.discover(force=True)

            self.assertNotIn("bad", h.registry.sessions)
            self.assertIn("good", h.registry.sessions)
            self.assertIn("invalid sidecar metadata", stderr.getvalue())
            self.assertIn("bad.sock", stderr.getvalue())

    def test_discovery_skips_bad_typed_sidecar_and_keeps_good_sidecar(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            bad_sock = sock_dir / "bad.sock"
            bad_sock.touch()
            bad_sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "bad",
                        "agent_backend": "codex",
                        "codex_pid": "0",
                        "broker_pid": 0,
                        "cwd": str(root),
                        "log_path": None,
                        "start_ts": 123.0,
                    }
                ),
                encoding="utf-8",
            )
            good_sock = sock_dir / "good.sock"
            good_sock.touch()
            _write_valid_sidecar(good_sock, root=root)
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                h.discover(force=True)

            self.assertNotIn("bad", h.registry.sessions)
            self.assertIn("good", h.registry.sessions)
            self.assertIn("invalid codex_pid", stderr.getvalue())

    def test_discovery_skips_bad_start_ts_without_pruning_sidecar(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            bad_sock = sock_dir / "bad.sock"
            bad_sock.touch()
            bad_sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "bad",
                        "agent_backend": "codex",
                        "codex_pid": 0,
                        "broker_pid": 0,
                        "cwd": str(root),
                        "log_path": None,
                        "start_ts": False,
                    }
                ),
                encoding="utf-8",
            )
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                h.discover(force=True)

            self.assertNotIn("bad", h.registry.sessions)
            self.assertTrue(bad_sock.exists())
            self.assertTrue(bad_sock.with_suffix(".json").exists())
            self.assertIn("invalid start_ts", stderr.getvalue())

    def test_discovery_rejects_nonfinite_start_ts_metadata(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            bad_sock = sock_dir / "bad.sock"
            bad_sock.touch()
            log_path = root / "bad.jsonl"
            log_path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "bad", "source": "cli"}}) + "\n", encoding="utf-8")
            bad_sock.with_suffix(".json").write_text(
                "{"
                '"session_id":"bad",'
                '"agent_backend":"codex",'
                '"codex_pid":0,'
                '"broker_pid":0,'
                f'"cwd":{json.dumps(str(root))},'
                f'"log_path":{json.dumps(str(log_path))},'
                '"start_ts":NaN'
                "}\n",
                encoding="utf-8",
            )
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                h.discover(force=True)

            self.assertNotIn("bad", h.registry.sessions)
            self.assertIn("invalid start_ts", stderr.getvalue())

    def test_discovery_rejects_overflowing_start_ts_metadata(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            bad_sock = sock_dir / "bad.sock"
            bad_sock.touch()
            log_path = root / "bad.jsonl"
            log_path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "bad", "source": "cli"}}) + "\n", encoding="utf-8")
            huge_int = "1" + ("0" * 400)
            bad_sock.with_suffix(".json").write_text(
                "{"
                '"session_id":"bad",'
                '"agent_backend":"codex",'
                '"codex_pid":0,'
                '"broker_pid":0,'
                f'"cwd":{json.dumps(str(root))},'
                f'"log_path":{json.dumps(str(log_path))},'
                f'"start_ts":{huge_int}'
                "}\n",
                encoding="utf-8",
            )
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                h.discover(force=True)

            self.assertNotIn("bad", h.registry.sessions)
            self.assertIn("invalid start_ts", stderr.getvalue())

    def test_discovery_skips_directory_log_path_and_keeps_good_sidecar(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            bad_sock = sock_dir / "bad.sock"
            bad_sock.touch()
            bad_sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "bad",
                        "agent_backend": "codex",
                        "codex_pid": 0,
                        "broker_pid": 0,
                        "cwd": str(root),
                        "log_path": str(root),
                        "start_ts": 123.0,
                    }
                ),
                encoding="utf-8",
            )
            good_sock = sock_dir / "good.sock"
            good_sock.touch()
            _write_valid_sidecar(good_sock, root=root)
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                h.discover(force=True)

            self.assertNotIn("bad", h.registry.sessions)
            self.assertIn("good", h.registry.sessions)
            self.assertIn("invalid log_path", stderr.getvalue())

    def test_discovery_tolerates_overflowing_optional_updated_ts(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            sock = sock_dir / "fixture.sock"
            sock.touch()
            log_path = root / "fixture.jsonl"
            log_path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "fixture", "source": "cli"}}) + "\n", encoding="utf-8")
            huge_int = "1" + ("0" * 400)
            sock.with_suffix(".json").write_text(
                "{"
                '"session_id":"fixture",'
                '"agent_backend":"codex",'
                '"codex_pid":0,'
                '"broker_pid":0,'
                f'"cwd":{json.dumps(str(root))},'
                f'"log_path":{json.dumps(str(log_path))},'
                '"start_ts":123.0,'
                f'"updated_ts":{huge_int}'
                "}\n",
                encoding="utf-8",
            )
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir)

            h.discover(force=True)

            self.assertIn("fixture", h.registry.sessions)
            self.assertIn(str(root), h.store.recent_cwds)
            self.assertTrue(math.isfinite(h.store.recent_cwds[str(root)]))

    def test_discovery_rejects_boolean_pid_metadata(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            bad_sock = sock_dir / "bad.sock"
            bad_sock.touch()
            log_path = root / "bad.jsonl"
            log_path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "bad", "source": "cli"}}) + "\n", encoding="utf-8")
            bad_sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "bad",
                        "agent_backend": "codex",
                        "codex_pid": True,
                        "broker_pid": 0,
                        "cwd": str(root),
                        "log_path": str(log_path),
                        "start_ts": 123.0,
                    }
                ),
                encoding="utf-8",
            )
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                h.discover(force=True)

            self.assertNotIn("bad", h.registry.sessions)
            self.assertIn("invalid codex_pid", stderr.getvalue())

    def test_refresh_skips_malformed_sidecar_and_keeps_existing_session(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock = root / "fixture.sock"
            sock.touch()
            sock.with_suffix(".json").write_text("[]\n", encoding="utf-8")
            sessions: dict = {"fixture": _session("fixture", sock)}
            existing = sessions["fixture"]
            existing.thread_id = "old-thread"
            existing.cwd = "/old"
            coord = _refresh_for_root(root, sessions=sessions)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                coord.refresh_session_meta("fixture")

            self.assertIs(sessions["fixture"], existing)
            self.assertEqual(existing.thread_id, "old-thread")
            self.assertEqual(existing.cwd, "/old")
            self.assertIn("invalid sidecar metadata", stderr.getvalue())

    def test_refresh_rejects_bad_typed_required_metadata_and_keeps_existing_session(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock = root / "fixture.sock"
            sock.touch()
            log_path = root / "fixture.jsonl"
            log_path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "new-thread", "source": "cli"}}) + "\n", encoding="utf-8")
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "new-thread",
                        "agent_backend": "codex",
                        "codex_pid": "bad",
                        "broker_pid": 0,
                        "cwd": str(root),
                        "log_path": str(log_path),
                        "start_ts": 123.0,
                    }
                ),
                encoding="utf-8",
            )
            sessions: dict = {"fixture": _session("fixture", sock)}
            existing = sessions["fixture"]
            existing.thread_id = "old-thread"
            existing.cwd = "/old"
            existing.log_path = None
            coord = _refresh_for_root(root, sessions=sessions)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                coord.refresh_session_meta("fixture")

            self.assertIs(sessions["fixture"], existing)
            self.assertEqual(existing.thread_id, "old-thread")
            self.assertEqual(existing.cwd, "/old")
            self.assertIsNone(existing.log_path)
            self.assertIn("invalid codex_pid", stderr.getvalue())

    def test_refresh_rejects_directory_log_path_and_keeps_existing_session(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock = root / "fixture.sock"
            sock.touch()
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "new-thread",
                        "agent_backend": "codex",
                        "codex_pid": 0,
                        "broker_pid": 0,
                        "cwd": str(root),
                        "log_path": str(root),
                        "start_ts": 123.0,
                    }
                ),
                encoding="utf-8",
            )
            sessions: dict = {"fixture": _session("fixture", sock)}
            existing = sessions["fixture"]
            existing.thread_id = "old-thread"
            existing.cwd = "/old"
            existing.log_path = None
            coord = _refresh_for_root(root, sessions=sessions)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                coord.refresh_session_meta("fixture")

            self.assertEqual(existing.thread_id, "old-thread")
            self.assertEqual(existing.cwd, "/old")
            self.assertIsNone(existing.log_path)
            self.assertIn("invalid log_path", stderr.getvalue())

    def test_refresh_rejects_nonfinite_start_ts_and_keeps_existing_session(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock = root / "fixture.sock"
            sock.touch()
            log_path = root / "fixture.jsonl"
            log_path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "new-thread", "source": "cli"}}) + "\n", encoding="utf-8")
            sock.with_suffix(".json").write_text(
                "{"
                '"session_id":"new-thread",'
                '"agent_backend":"codex",'
                '"codex_pid":0,'
                '"broker_pid":0,'
                f'"cwd":{json.dumps(str(root))},'
                f'"log_path":{json.dumps(str(log_path))},'
                '"start_ts":Infinity'
                "}\n",
                encoding="utf-8",
            )
            sessions: dict = {"fixture": _session("fixture", sock)}
            existing = sessions["fixture"]
            existing.thread_id = "old-thread"
            existing.cwd = "/old"
            existing.log_path = None
            coord = _refresh_for_root(root, sessions=sessions)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                coord.refresh_session_meta("fixture")

            self.assertEqual(existing.thread_id, "old-thread")
            self.assertEqual(existing.cwd, "/old")
            self.assertIsNone(existing.log_path)
            self.assertIn("invalid start_ts", stderr.getvalue())

    def test_refresh_rejects_overflowing_start_ts_and_keeps_existing_session(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock = root / "fixture.sock"
            sock.touch()
            log_path = root / "fixture.jsonl"
            log_path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "new-thread", "source": "cli"}}) + "\n", encoding="utf-8")
            huge_int = "1" + ("0" * 400)
            sock.with_suffix(".json").write_text(
                "{"
                '"session_id":"new-thread",'
                '"agent_backend":"codex",'
                '"codex_pid":0,'
                '"broker_pid":0,'
                f'"cwd":{json.dumps(str(root))},'
                f'"log_path":{json.dumps(str(log_path))},'
                f'"start_ts":{huge_int}'
                "}\n",
                encoding="utf-8",
            )
            sessions: dict = {"fixture": _session("fixture", sock)}
            existing = sessions["fixture"]
            existing.thread_id = "old-thread"
            existing.cwd = "/old"
            existing.log_path = None
            coord = _refresh_for_root(root, sessions=sessions)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                coord.refresh_session_meta("fixture")

            self.assertEqual(existing.thread_id, "old-thread")
            self.assertEqual(existing.cwd, "/old")
            self.assertIsNone(existing.log_path)
            self.assertIn("invalid start_ts", stderr.getvalue())

    def test_discovery_keeps_sidecar_log_without_codex_session_metadata(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            sock = sock_dir / "fixture.sock"
            sock.touch()
            log_path = root / "rollout-no-session-meta.jsonl"
            log_path.write_text(
                json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}}) + "\n",
                encoding="utf-8",
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "sidecar-thread",
                        "agent_backend": "codex",
                        "codex_pid": 0,
                        "broker_pid": 0,
                        "cwd": str(root),
                        "log_path": str(log_path),
                        "start_ts": 123.0,
                    }
                ),
                encoding="utf-8",
            )
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir)

            h.discover(force=True)

            self.assertIn("fixture", h.registry.sessions)
            self.assertEqual(h.registry.sessions["fixture"].thread_id, "sidecar-thread")
            self.assertEqual(h.registry.sessions["fixture"].log_path, log_path)

    def test_discovery_skips_malformed_broker_state_without_coercion(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            sock = sock_dir / "fixture.sock"
            sock.touch()
            log_path = root / "rollout.jsonl"
            log_path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "sidecar-thread", "source": "cli"}}) + "\n", encoding="utf-8")
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "sidecar-thread",
                        "agent_backend": "codex",
                        "codex_pid": 0,
                        "broker_pid": 0,
                        "cwd": str(root),
                        "log_path": str(log_path),
                        "start_ts": 123.0,
                    }
                ),
                encoding="utf-8",
            )
            h = _DiscoveryHarness(root=root, sock_dir=sock_dir)

            # Inject a broker control-socket response whose ``busy`` field is a
            # string; the real ``broker_busy_queue`` parser must reject it
            # (socket/IPC response boundary).
            original_deps = h._discovery_deps
            h._discovery_deps = lambda: DiscoveryDeps(  # type: ignore[method-assign]
                pid_alive=_pid_alive_impl,
                proc_find_open_rollout_log=_proc_find_open_rollout_log_impl,
                read_session_meta_or_none=_read_session_meta_or_none,
                coerce_main_thread_log=_coerce_main_thread_log,
                session_transport=lambda meta: session_transport_from_meta(meta=meta, clean_optional_text=_clean_optional_text),
                session_run_settings=lambda meta, log_path, agent_backend: session_run_settings_from_meta(
                    meta=meta,
                    log_path=log_path,
                    agent_backend=agent_backend,
                    clean_optional_text=_clean_optional_text,
                    normalize_requested_preferred_auth_method=server._normalize_requested_preferred_auth_method,
                    display_reasoning_effort=server._display_reasoning_effort,
                    display_pi_reasoning_effort=server._display_pi_reasoning_effort,
                    normalize_requested_cc_reasoning_effort=server._normalize_requested_cc_reasoning_effort,
                    read_run_settings_from_log=server._read_run_settings_from_log,
                ),
                sock_call=lambda _sock, _req, _timeout_s: {"busy": "false", "queue_len": 0, "token": None},
                broker_busy_queue_from_state=broker_busy_queue,
                broker_interrupted_idle_from_state=broker_interrupted_idle,
                sock_error_definitely_stale=lambda _exc: False,
                token_update_finder=lambda _log_path: None,
            )
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr):
                    h.discover(force=True)
            finally:
                h._discovery_deps = original_deps  # type: ignore[method-assign]

            self.assertNotIn("fixture", h.registry.sessions)
            self.assertIn("invalid broker state", stderr.getvalue())

    def test_invalid_session_meta_warning_is_once_per_context_path(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout-no-session-meta.jsonl"
            log_path.write_text(
                json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}}) + "\n",
                encoding="utf-8",
            )
            # Locally-owned dedup set + stderr stream replace the former
            # ``server._INVALID_SESSION_META_WARNINGS`` module-global mutation.
            invalid_warnings: set = set()
            stderr = io.StringIO()

            first = _read_session_meta_or_none_impl(
                log_path,
                agent_backend="codex",
                context="test",
                read_session_meta_func=server._read_session_meta,
                invalid_warnings=invalid_warnings,
                stderr=stderr,
            )
            second = _read_session_meta_or_none_impl(
                log_path,
                agent_backend="codex",
                context="test",
                read_session_meta_func=server._read_session_meta,
                invalid_warnings=invalid_warnings,
                stderr=stderr,
            )

            self.assertIsNone(first)
            self.assertIsNone(second)
            self.assertEqual(stderr.getvalue().count("ignoring invalid session metadata"), 1)

    def test_refresh_keeps_sidecar_log_without_codex_session_metadata(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock = root / "fixture.sock"
            sock.touch()
            log_path = root / "rollout-no-session-meta.jsonl"
            log_path.write_text(
                json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "hello"}}) + "\n",
                encoding="utf-8",
            )
            sock.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": "sidecar-thread",
                        "agent_backend": "codex",
                        "codex_pid": 0,
                        "broker_pid": 0,
                        "cwd": str(root),
                        "log_path": str(log_path),
                        "start_ts": 123.0,
                    }
                ),
                encoding="utf-8",
            )
            sessions: dict = {"fixture": _session("fixture", sock)}
            coord = _refresh_for_root(root, sessions=sessions)

            coord.refresh_session_meta("fixture")

            self.assertIn("fixture", sessions)
            self.assertEqual(sessions["fixture"].thread_id, "sidecar-thread")
            self.assertEqual(sessions["fixture"].log_path, log_path)


if __name__ == "__main__":
    unittest.main()
