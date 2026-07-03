"""Direct coordinator/dependency-injection tests for session resume logic.

Previously these tests patched ``codoxear.server`` internals (~38 sites):
``SOCK_DIR``, ``_iter_session_logs``, ``_wait_or_raise``, ``subprocess.Popen``,
``_list_resume_candidates_for_cwd``, ``_create_git_worktree``, ``_pid_alive``,
``_proc_find_open_rollout_log``, ``shutil.which``, ``subprocess.run``,
``_wait_for_spawned_broker_meta``, ``_tmux_pane_snapshot`` and constructed
``SessionManager`` via ``__new__``.

They now exercise the true seams directly:

* resume candidate listing / preview -> ``codoxear.session_resume`` free
  functions with ``iter_session_logs`` / ``pi_user_text`` / ``cc_user_text``
  injected as callables (no module-global patch).
* spawned-broker metadata polling -> ``codoxear.session_launcher.
  wait_for_spawned_broker_meta`` with ``sock_dir`` passed directly.
* web-session spawn argv/env contract -> ``SessionWebLaunchCoordinator`` built
  with the real launch-plan builders and injected fakes for the
  process/socket/tmux boundaries.
* rollout delivery suppression during resume catch-up ->
  ``VoiceRuntimeCoordinator.observe_rollout_delta`` with injected deps.
* log binding / detach on refresh and discovery -> ``SessionRefreshCoordinator``
  and ``discover_sessions`` with injected ``proc_find_open_rollout_log`` /
  ``sock_call`` / ``pid_alive``.

No ``codoxear.server.*`` module-global monkeypatching remains. The only
retained ``patch`` calls target genuine OS/process/socket/network boundaries
(``pathlib.Path.mkdir`` to simulate a permission failure and
``threading.Thread.start`` to avoid spawning real reaper threads); each carries
an explicit justification at its use site. No file under ``codoxear/`` is
modified.
"""

import json
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY
from unittest.mock import patch

from codoxear import git_ops
from codoxear import server
from codoxear.cc_log import cc_user_text
from codoxear.launch_path_runtime import codex_trust_override_for_path as _codex_trust_override_for_path_impl
from codoxear.launch_path_runtime import load_env_file as _load_env_file_impl
from codoxear.launch_path_runtime import resolve_dir_target as _resolve_dir_target_impl
from codoxear.pi_log import pi_user_text
from codoxear.rollout_delivery import _extract_delivery_messages
from codoxear.rollout_log import _cc_pending_tool_ids_before
from codoxear.session_discovery import DiscoveryDeps
from codoxear.session_discovery import discover_sessions
from codoxear.session_errors import SessionLaunchError
from codoxear.session_launcher import drain_stream as _drain_stream_impl
from codoxear.session_launcher import wait_for_spawned_broker_meta as _wait_for_spawned_broker_meta_impl
from codoxear.session_lifecycle import SessionLifecycleCoordinator
from codoxear.session_model import Session
from codoxear.session_refresh import SessionRefreshCoordinator
from codoxear.session_refresh import broker_tail_has_session_detach_marker
from codoxear.session_resume import first_user_message_preview_from_log as _first_user_message_preview_from_log_impl
from codoxear.session_resume import list_resume_candidates_for_cwd as _list_resume_candidates_for_cwd_impl
from codoxear.session_resume import resume_candidate_from_log as _resume_candidate_from_log_impl
from codoxear.session_runtime import broker_busy_queue
from codoxear.session_runtime import broker_interrupted_idle
from codoxear.session_runtime import reset_session_log_caches
from codoxear.session_web_launch import SessionWebLaunchCoordinator
from codoxear.sidecar_metadata import _clean_optional_text as _sidecar_clean_optional_text
from codoxear.util import is_subagent_session_meta
from codoxear.voice_runtime import VoiceRuntimeCoordinator

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(obj) + "\n" for obj in objs), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Resume-candidate helpers: real impls wired with injected callables.
# --------------------------------------------------------------------------- #


def _resume_candidate(log_path: Path, *, agent_backend: str) -> dict | None:
    """Real ``resume_candidate_from_log`` wired with the server-configured
    ``read_session_meta`` and the shared ``is_subagent_session_meta``."""
    return _resume_candidate_from_log_impl(
        log_path,
        agent_backend=agent_backend,
        read_session_meta=server._read_session_meta,
        is_subagent_session_meta=is_subagent_session_meta,
    )


def _list_resume_candidates(cwd: str, logs: list[Path], *, agent_backend: str = "codex", limit: int = 12) -> list[dict]:
    """``list_resume_candidates_for_cwd`` with ``iter_session_logs`` injected so
    the test controls which logs are seen, instead of patching the module
    global ``codoxear.server._iter_session_logs``."""
    return _list_resume_candidates_for_cwd_impl(
        cwd,
        agent_backend=agent_backend,
        limit=limit,
        iter_session_logs=lambda **_kw: list(logs),
        resume_candidate_from_log_func=_resume_candidate,
    )


def _first_user_message_preview(log_path: Path) -> str:
    """Real preview builder with backend text extractors injected (the server
    facade closes over module globals; wiring the impl keeps the seam explicit)."""
    return _first_user_message_preview_from_log_impl(
        log_path,
        pi_user_text=pi_user_text,
        cc_user_text=cc_user_text,
    )


# --------------------------------------------------------------------------- #
# Web-launch coordinator builder: real plan builders + injected boundaries.
# --------------------------------------------------------------------------- #


class _FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.stderr = None

    def wait(self) -> int:
        return 0


class _PopenRecorder:
    """Captures argv/env of the single broker spawn so spawn-contract tests can
    assert on them without patching ``codoxear.server.subprocess.Popen``."""

    def __init__(self, proc: _FakeProc) -> None:
        self._proc = proc
        self.argv: list[str] | None = None
        self.env: dict[str, str] | None = None
        self.calls = 0

    def __call__(self, argv: list[str], **kwargs: object) -> _FakeProc:
        self.argv = list(argv)
        self.env = dict(kwargs.get("env") or {})
        self.calls += 1
        return self._proc


def _web_launch_coordinator(**overrides) -> SessionWebLaunchCoordinator:
    """``SessionWebLaunchCoordinator`` wired with the real launch-plan builders
    and the real backend-environment/trust-override logic. Process/socket/tmux
    boundaries default to inert fakes and are overridden per test."""
    defaults = dict(
        resolve_dir_target=_resolve_dir_target_impl,
        create_git_worktree=server._create_git_worktree,
        codex_trust_override_for_path=_codex_trust_override_for_path_impl,
        list_resume_candidates_for_cwd=lambda cwd, *, agent_backend, limit: _list_resume_candidates_for_cwd_impl(
            cwd,
            agent_backend=agent_backend,
            limit=limit,
            iter_session_logs=server._iter_session_logs,
            resume_candidate_from_log_func=_resume_candidate,
        ),
        live_session_for_resume_target=lambda _resume_id, _row: None,
        load_env_file=_load_env_file_impl,
        environ=dict(os.environ),
        dotenv_path=Path("/nonexistent-codoxear-dotenv-test"),
        homes={"codex": server.CODEX_HOME, "pi": server.PI_HOME, "cc": server.CC_HOME},
        python_executable=sys.executable,
        tmux_session_name="codoxear",
        repo_root=REPO_ROOT,
        record_launch_attempt=lambda record: record,
        now=time.time,
        stderr=None,
        which_tmux=lambda _name: None,
        run=subprocess.run,
        popen=subprocess.Popen,
        wait_or_raise=server._wait_or_raise,
        wait_for_spawned_broker_meta=lambda _nonce: {"broker_pid": 0},
        tmux_pane_snapshot=lambda *a, **kw: {},
        drain_stream=lambda _stream: None,
        launch_error=SessionLaunchError,
    )
    defaults.update(overrides)
    return SessionWebLaunchCoordinator(**defaults)


def _capture_thread_starts() -> tuple[list[str], object]:
    """Records ``threading.Thread.start`` invocations.

    Residual OS/process-boundary patch: ``launch_broker_process`` starts real
    daemon threads to reap the broker process and drain its stderr. Letting
    those run in tests would spawn live threads against fake proc objects and
    leak nondeterministically, so ``Thread.start`` is intercepted exactly as in
    the sibling ``tests/test_session_launcher.py`` reference."""
    starts: list[str] = []

    def _start(self) -> None:
        starts.append("start")

    return starts, _start


# --------------------------------------------------------------------------- #
# Refresh / discovery coordinator builders.
# --------------------------------------------------------------------------- #


def _read_session_meta_or_none_adapted(log_path, *args, **kwargs):
    """Adapter accepting both the discovery (positional) and refresh (keyword)
    calling conventions, delegating to the real server-configured reader."""
    agent_backend = kwargs.get("agent_backend")
    if agent_backend is None and args:
        agent_backend = args[0]
    context = kwargs.get("context")
    if context is None and len(args) >= 2:
        context = args[1]
    if context is None:
        context = "discovery"
    return server._read_session_meta_or_none(log_path, agent_backend=agent_backend, context=context)


def _refresh_coordinator(
    *,
    sessions: dict[str, Session],
    proc_root: Path,
    proc_find_open_rollout_log,
    sock_call,
    broker_tail_has_session_detach_marker=broker_tail_has_session_detach_marker,
    pid_alive=lambda _pid: True,
    read_session_meta_or_none=_read_session_meta_or_none_adapted,
    coerce_main_thread_log=lambda **kw: (kw["thread_id"], kw["log_path"]),
) -> SessionRefreshCoordinator:
    return SessionRefreshCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        prune_stale_socket_without_metadata=lambda _sid, _sock: None,
        log_invalid_sidecar_metadata=lambda _ctx, _sock, _exc: None,
        session_transport=lambda **_kw: (None, None, None),
        sock_call=sock_call,
        broker_tail_has_session_detach_marker=broker_tail_has_session_detach_marker,
        pid_alive=pid_alive,
        proc_find_open_rollout_log=proc_find_open_rollout_log,
        proc_root=proc_root,
        read_session_meta_or_none=read_session_meta_or_none,
        coerce_main_thread_log=coerce_main_thread_log,
        clean_optional_text=_sidecar_clean_optional_text,
        session_run_settings=lambda **_kw: (None, None, None, None),
        normalize_requested_service_tier=lambda _value: None,
        reset_log_caches=lambda session, log_off: reset_session_log_caches(session, meta_log_off=log_off),
        queue_len=lambda _sid: 0,
        maybe_drain_session_queue=lambda _sid: None,
    )


def _discovery_deps(
    *,
    proc_find_open_rollout_log,
    pid_alive=lambda _pid: True,
    sock_call=lambda *a, **kw: {"busy": False, "queue_len": 0, "token": None},
    read_session_meta_or_none=_read_session_meta_or_none_adapted,
    coerce_main_thread_log=lambda thread_id, log_path: (thread_id, log_path),
) -> DiscoveryDeps:
    return DiscoveryDeps(
        pid_alive=pid_alive,
        proc_find_open_rollout_log=proc_find_open_rollout_log,
        read_session_meta_or_none=read_session_meta_or_none,
        coerce_main_thread_log=coerce_main_thread_log,
        session_transport=lambda _meta: (None, None, None),
        session_run_settings=lambda *_a, **_kw: (None, None, None, None),
        sock_call=sock_call,
        broker_busy_queue_from_state=broker_busy_queue,
        broker_interrupted_idle_from_state=broker_interrupted_idle,
        sock_error_definitely_stale=lambda _exc: False,
        token_update_finder=lambda _path: None,
    )


def _registration_to_session(reg) -> Session:
    return Session(
        session_id=reg.session_id,
        thread_id=reg.thread_id,
        broker_pid=reg.broker_pid,
        codex_pid=reg.codex_pid,
        agent_backend=reg.agent_backend,
        owned=reg.owned,
        start_ts=reg.start_ts,
        cwd=reg.cwd,
        log_path=reg.log_path,
        sock_path=reg.sock_path,
        busy=reg.busy,
        queue_len=reg.queue_len,
        token=reg.token,
        meta_log_off=reg.meta_log_off,
        model_provider=reg.model_provider,
        preferred_auth_method=reg.preferred_auth_method,
        model=reg.model,
        reasoning_effort=reg.reasoning_effort,
        service_tier=reg.service_tier,
        tmux_session=reg.tmux_session,
        tmux_window=reg.tmux_window,
        launch_id=reg.launch_id,
        spawn_nonce=reg.spawn_nonce,
        resume_session_id=reg.resume_session_id,
        sync_send_supported=reg.sync_send_supported,
        key_write_errors_supported=reg.key_write_errors_supported,
        interrupted_idle=reg.interrupted_idle,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestSessionResumeCandidates(unittest.TestCase):
    def test_wait_for_spawned_broker_meta_skips_bool_broker_pid(self) -> None:
        # sock_dir is injected directly into wait_for_spawned_broker_meta instead
        # of patching the module-global codoxear.server.SOCK_DIR.
        with TemporaryDirectory() as td:
            sock_dir = Path(td)
            bad = sock_dir / "a-bad.json"
            bad.write_text(json.dumps({"spawn_nonce": "nonce", "broker_pid": True}) + "\n", encoding="utf-8")
            good = sock_dir / "b-good.json"
            good.write_text(json.dumps({"spawn_nonce": "nonce", "broker_pid": os.getpid()}) + "\n", encoding="utf-8")

            meta = _wait_for_spawned_broker_meta_impl("nonce", sock_dir=sock_dir, timeout_s=0.01)

        self.assertEqual(meta["broker_pid"], os.getpid())

    def test_wait_for_spawned_broker_meta_ignores_malformed_json(self) -> None:
        with TemporaryDirectory() as td:
            sock_dir = Path(td)
            bad = sock_dir / "bad.json"
            bad.write_text("{not-json}\n", encoding="utf-8")
            good = sock_dir / "good.json"
            good.write_text(json.dumps({"spawn_nonce": "nonce", "broker_pid": os.getpid()}) + "\n", encoding="utf-8")

            meta = _wait_for_spawned_broker_meta_impl("nonce", sock_dir=sock_dir, timeout_s=0.01)

        self.assertEqual(meta["broker_pid"], os.getpid())

    def test_wait_for_spawned_broker_meta_skips_nonlive_broker_pid(self) -> None:
        with TemporaryDirectory() as td:
            sock_dir = Path(td)
            zero = sock_dir / "a-zero.json"
            zero.write_text(json.dumps({"spawn_nonce": "nonce", "broker_pid": 0}) + "\n", encoding="utf-8")
            negative = sock_dir / "b-negative.json"
            negative.write_text(json.dumps({"spawn_nonce": "nonce", "broker_pid": -1}) + "\n", encoding="utf-8")
            dead = sock_dir / "c-dead.json"
            dead.write_text(json.dumps({"spawn_nonce": "nonce", "broker_pid": 2147483647}) + "\n", encoding="utf-8")
            good = sock_dir / "d-good.json"
            good.write_text(json.dumps({"spawn_nonce": "nonce", "broker_pid": os.getpid()}) + "\n", encoding="utf-8")

            meta = _wait_for_spawned_broker_meta_impl("nonce", sock_dir=sock_dir, timeout_s=0.01)

        self.assertEqual(meta["broker_pid"], os.getpid())

    def test_describe_session_cwd_marks_missing_dir_for_creation(self) -> None:
        # _describe_session_cwd is a pure read-only facade over the real git
        # helpers; for a missing target no git probe runs, so it is exercised
        # directly (no module-global patch and no closure bypass).
        from codoxear.client_file_paths import describe_session_cwd

        with TemporaryDirectory() as td:
            target = Path(td) / "missing" / "child"
            info = describe_session_cwd(
                target.resolve(),
                git_repo_root=server._git_repo_root,
                current_git_branch=server._current_git_branch,
            )

        self.assertEqual(info["cwd"], str(target.resolve()))
        self.assertFalse(info["exists"])
        self.assertTrue(info["will_create"])
        self.assertFalse(info["git_repo"])

    def test_list_resume_candidates_filters_same_cwd_and_dedupes(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            same_new = root / "rollout-2026-03-08T01-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            same_old = root / "rollout-2026-03-07T01-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            other = root / "rollout-2026-03-08T02-00-00-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            child = root / "rollout-2026-03-08T03-00-00-cccccccc-cccc-cccc-cccc-cccccccccccc.jsonl"

            _write_jsonl(same_new, [{"type": "session_meta", "payload": {"id": "resume-a", "cwd": "/repo", "timestamp": "2026-03-08T01:00:00Z", "source": "cli"}}])
            _write_jsonl(same_old, [{"type": "session_meta", "payload": {"id": "resume-a", "cwd": "/repo", "timestamp": "2026-03-07T01:00:00Z", "source": "cli"}}])
            _write_jsonl(other, [{"type": "session_meta", "payload": {"id": "resume-b", "cwd": "/elsewhere", "timestamp": "2026-03-08T02:00:00Z", "source": "cli"}}])
            _write_jsonl(
                child,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "resume-child",
                            "cwd": "/repo",
                            "source": {"subagent": {"thread_spawn": {"parent_thread_id": "resume-a", "depth": 1}}},
                        },
                    }
                ],
            )

            rows = _list_resume_candidates("/repo", [same_new, child, other, same_old], limit=10)

        self.assertEqual([row["session_id"] for row in rows], ["resume-a"])
        self.assertEqual(rows[0]["cwd"], "/repo")
        self.assertEqual(rows[0]["log_path"], str(same_new))

    def test_first_user_message_preview_skips_unattended_scaffolding(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout-2026-03-08T01-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            _write_jsonl(
                log_path,
                [
                    {"type": "session_meta", "payload": {"id": "resume-a", "cwd": "/repo", "source": "cli"}},
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "# AGENTS.md instructions for /repo\n..."},
                                {"type": "input_text", "text": "<environment_context>\n  <cwd>/repo</cwd>\n</environment_context>"},
                            ],
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Is it possible to extract something like the conversation title or at least the first user message?",
                                }
                            ],
                        },
                    },
                ],
            )

            preview = _first_user_message_preview(log_path)

        self.assertEqual(
            preview,
            "Is it possible to extract something like the conversation title or at least the first user message?",
        )

    def test_list_cc_resume_candidates_filters_same_cwd(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            same = root / "11111111-2222-3333-4444-555555555555.jsonl"
            other = root / "22222222-2222-3333-4444-555555555555.jsonl"
            _write_jsonl(
                same,
                [
                    {
                        "type": "user",
                        "sessionId": "cc-a",
                        "cwd": "/repo",
                        "timestamp": "2026-03-08T01:00:00Z",
                        "gitBranch": "feature/cc",
                        "message": {"role": "user", "content": "hello from cc"},
                    }
                ],
            )
            _write_jsonl(
                other,
                [
                    {
                        "type": "user",
                        "sessionId": "cc-b",
                        "cwd": "/elsewhere",
                        "timestamp": "2026-03-08T02:00:00Z",
                        "message": {"role": "user", "content": "hello elsewhere"},
                    }
                ],
            )

            rows = _list_resume_candidates("/repo", [same, other], agent_backend="cc", limit=10)

            self.assertEqual([row["session_id"] for row in rows], ["cc-a"])
            self.assertEqual(rows[0]["agent_backend"], "cc")
            self.assertEqual(rows[0]["git_branch"], "feature/cc")
            self.assertEqual(rows[0]["log_path"], str(same))
            self.assertEqual(_first_user_message_preview(same), "hello from cc")

    def test_list_pi_resume_candidates_filters_same_cwd(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            same = root / "2026-03-08T01-00-00_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            other = root / "2026-03-08T02-00-00_bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            _write_jsonl(
                same,
                [
                    {"type": "session", "id": "pi-a", "cwd": "/repo", "timestamp": "2026-03-08T01:00:00Z"},
                    {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hello from pi"}]}},
                ],
            )
            _write_jsonl(other, [{"type": "session", "id": "pi-b", "cwd": "/elsewhere", "timestamp": "2026-03-08T02:00:00Z"}])

            rows = _list_resume_candidates("/repo", [same, other], agent_backend="pi", limit=10)

        self.assertEqual([row["session_id"] for row in rows], ["pi-a"])
        self.assertEqual(rows[0]["agent_backend"], "pi")
        self.assertEqual(rows[0]["log_path"], str(same))


class _FakeVoicePush:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def observe_messages(self, *, session_id: str, session_display_name: str, messages: list[object]) -> None:
        self.calls.append(
            {
                "session_id": session_id,
                "session_display_name": session_display_name,
                "messages": messages,
            }
        )


def _voice_coordinator(*, sessions: dict[str, Session], voice_push) -> VoiceRuntimeCoordinator:
    return VoiceRuntimeCoordinator(
        lock=threading.Lock(),
        sessions=lambda: sessions,
        aliases=lambda: {},
        voice_push=lambda: voice_push,
        discover_existing_if_stale=lambda: None,
        prune_dead_sessions=lambda: None,
        refresh_session_meta=lambda _sid: None,
        read_jsonl_from_offset=lambda _path, _off, **_kw: ([], 0),
        extract_delivery_messages=_extract_delivery_messages,
        cc_pending_tool_ids_before=_cc_pending_tool_ids_before,
    )


class TestSpawnWebSessionResume(unittest.TestCase):
    def test_spawn_web_session_creates_missing_cwd(self) -> None:
        thread_starts, start_fn = _capture_thread_starts()
        popen = _PopenRecorder(_FakeProc(2468))
        coordinator = _web_launch_coordinator(
            wait_or_raise=lambda *a, **kw: None,
            popen=popen,
        )
        with TemporaryDirectory() as td, patch.object(threading.Thread, "start", start_fn):
            target = Path(td) / "new" / "session"
            result = coordinator.spawn_web_session(cwd=str(target))
            self.assertTrue(target.is_dir())

        trust_override = f'projects={{ {json.dumps(str(target.resolve()))} = {{ trust_level = "trusted" }} }}'
        self.assertEqual(
            popen.argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                str(target.resolve()),
                "--",
                "-c",
                trust_override,
                "-c",
                "check_for_update_on_startup=false",
                "--disable",
                "goals",
                "--dangerously-bypass-approvals-and-sandbox",
            ],
        )
        self.assertEqual(result, {"broker_pid": 2468})
        self.assertEqual(thread_starts, ["start"])

    def test_spawn_web_session_surfaces_mkdir_failure(self) -> None:
        coordinator = _web_launch_coordinator()
        with TemporaryDirectory() as td, patch("pathlib.Path.mkdir", side_effect=PermissionError(13, "Permission denied")):
            target = Path(td) / "blocked" / "session"
            with self.assertRaisesRegex(ValueError, r"cwd could not be created: .*Permission denied"):
                coordinator.spawn_web_session(cwd=str(target))

    def test_spawn_web_session_marks_spawn_cwd_trusted(self) -> None:
        thread_starts, start_fn = _capture_thread_starts()
        popen = _PopenRecorder(_FakeProc(3210))
        coordinator = _web_launch_coordinator(
            wait_or_raise=lambda *a, **kw: None,
            popen=popen,
        )
        with TemporaryDirectory() as td, patch.object(threading.Thread, "start", start_fn):
            result = coordinator.spawn_web_session(cwd=td, args=["--search"])

        trust_override = f'projects={{ {json.dumps(str(Path(td).resolve()))} = {{ trust_level = "trusted" }} }}'
        self.assertEqual(
            popen.argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                td,
                "--",
                "-c",
                trust_override,
                "-c",
                "check_for_update_on_startup=false",
                "--disable",
                "goals",
                "--dangerously-bypass-approvals-and-sandbox",
                "--search",
            ],
        )
        self.assertEqual(result, {"broker_pid": 3210})
        self.assertEqual(thread_starts, ["start"])

    def test_spawn_web_session_passes_resume_id_to_broker(self) -> None:
        thread_starts, start_fn = _capture_thread_starts()
        popen = _PopenRecorder(_FakeProc(4321))
        coordinator = _web_launch_coordinator(
            list_resume_candidates_for_cwd=lambda cwd, *, agent_backend, limit: [{"session_id": "resume-a"}],
            wait_or_raise=lambda *a, **kw: None,
            popen=popen,
        )
        with TemporaryDirectory() as td, patch.object(threading.Thread, "start", start_fn):
            result = coordinator.spawn_web_session(
                cwd=td,
                args=["--search"],
                resume_session_id="resume-a",
            )

        trust_override = f'projects={{ {json.dumps(str(Path(td).resolve()))} = {{ trust_level = "trusted" }} }}'
        self.assertEqual(
            popen.argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                td,
                "--",
                "-c",
                trust_override,
                "-c",
                "check_for_update_on_startup=false",
                "--disable",
                "goals",
                "--dangerously-bypass-approvals-and-sandbox",
                "resume",
                "resume-a",
                "--search",
            ],
        )
        self.assertEqual(popen.env["CODEX_WEB_RESUME_SESSION_ID"], "resume-a")
        self.assertEqual(result, {"broker_pid": 4321})
        self.assertEqual(thread_starts, ["start"])

    def test_spawn_web_session_rejects_resume_target_that_is_already_live(self) -> None:
        with TemporaryDirectory() as td:
            log_path = Path(td) / "rollout-2026-04-26T01-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            live_session = Session(
                session_id="live-row",
                thread_id="resume-a",
                broker_pid=111,
                codex_pid=222,
                agent_backend="codex",
                owned=True,
                start_ts=1.0,
                cwd=td,
                log_path=log_path,
                sock_path=Path(td) / "live.sock",
            )
            lifecycle = SessionLifecycleCoordinator(
                lock=threading.Lock(),
                sessions=lambda: {live_session.session_id: live_session},
                sock_call=lambda *a, **kw: {},
                process_group_alive=lambda _pgid: False,
                pid_alive=lambda _pid: True,
                terminate_process_group=lambda *a, **kw: True,
                terminate_process=lambda *a, **kw: True,
                unlink_quiet=lambda _p: None,
                commit_unknown_sends=lambda: {},
                queue_has_recovery_items_locked=lambda _sid: False,
                clear_deleted_session_state=lambda *a, **kw: None,
                read_launch_attempts=lambda: [],
                launch_attempt_row=lambda _r: None,
                hide_session=lambda _sid: None,
            )

            def _popen_no_call(*a, **kw):
                raise AssertionError("Popen must not be called when the resume target is already live")

            coordinator = _web_launch_coordinator(
                list_resume_candidates_for_cwd=lambda cwd, *, agent_backend, limit: [
                    {"session_id": "resume-a", "log_path": str(log_path)}
                ],
                live_session_for_resume_target=lifecycle.live_session_for_resume_target,
                popen=_popen_no_call,
            )
            with self.assertRaisesRegex(ValueError, "resume target is already live as live-row"):
                coordinator.spawn_web_session(cwd=td, resume_session_id="resume-a")

    def test_spawn_web_session_passes_model_and_reasoning_to_broker(self) -> None:
        thread_starts, start_fn = _capture_thread_starts()
        popen = _PopenRecorder(_FakeProc(6543))
        coordinator = _web_launch_coordinator(
            wait_or_raise=lambda *a, **kw: None,
            popen=popen,
        )
        with TemporaryDirectory() as td, patch.object(threading.Thread, "start", start_fn):
            result = coordinator.spawn_web_session(
                cwd=td,
                model_provider="bytecat",
                preferred_auth_method="apikey",
                model="gpt-5.4",
                reasoning_effort="xhigh",
                service_tier="fast",
            )

        trust_override = f'projects={{ {json.dumps(str(Path(td).resolve()))} = {{ trust_level = "trusted" }} }}'
        self.assertEqual(
            popen.argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                td,
                "--",
                "-c",
                trust_override,
                "-c",
                "check_for_update_on_startup=false",
                "--disable",
                "goals",
                "--dangerously-bypass-approvals-and-sandbox",
                "--model",
                "gpt-5.4",
                "-c",
                'model_reasoning_effort="xhigh"',
                "-c",
                'model_provider="bytecat"',
                "-c",
                'preferred_auth_method="apikey"',
                "-c",
                'service_tier="fast"',
            ],
        )
        self.assertEqual(popen.env["CODEX_WEB_MODEL_PROVIDER"], "bytecat")
        self.assertEqual(popen.env["CODEX_WEB_PREFERRED_AUTH_METHOD"], "apikey")
        self.assertEqual(popen.env["CODEX_WEB_MODEL"], "gpt-5.4")
        self.assertEqual(popen.env["CODEX_WEB_REASONING_EFFORT"], "xhigh")
        self.assertEqual(popen.env["CODEX_WEB_SERVICE_TIER"], "fast")
        self.assertEqual(result, {"broker_pid": 6543})
        self.assertEqual(thread_starts, ["start"])

    def test_spawn_web_session_can_start_in_tmux(self) -> None:
        run_calls: list[list[str]] = []
        side_effects = [
            subprocess.CompletedProcess(["/usr/bin/tmux", "new-window"], 1, stdout="", stderr="error connecting to /tmp/tmux-0/default (No such file or directory)"),
            subprocess.CompletedProcess(["/usr/bin/tmux", "new-session"], 0, stdout="%8\n", stderr=""),
        ]

        def fake_run(argv, **kwargs):
            run_calls.append(list(argv))
            return side_effects.pop(0)

        coordinator = _web_launch_coordinator(
            which_tmux=lambda _name: "/usr/bin/tmux",
            wait_for_spawned_broker_meta=lambda _nonce: {"broker_pid": 7777},
            tmux_pane_snapshot=lambda *a, **kw: {"tmux_pane_id": "%8", "tmux_pane_dead": "0", "tmux_window": "work-abc123"},
            run=fake_run,
        )
        with TemporaryDirectory() as td:
            result = coordinator.spawn_web_session(
                cwd=td,
                model_provider="crs",
                preferred_auth_method="apikey",
                model="gpt-5.4",
                service_tier="fast",
                create_in_tmux=True,
            )

        self.assertEqual(result, {"broker_pid": 7777, "tmux_session": "codoxear", "tmux_window": ANY})
        tmux_argv = run_calls[1]
        self.assertEqual(tmux_argv[:8], ["/usr/bin/tmux", "new-session", "-d", "-P", "-F", "#{pane_id}", "-s", "codoxear"])
        shell_cmd = tmux_argv[-1]
        self.assertIn("CODEX_WEB_TRANSPORT=tmux", shell_cmd)
        self.assertIn("CODEX_WEB_TMUX_SESSION=codoxear", shell_cmd)
        self.assertIn("CODEX_WEB_TMUX_WINDOW=", shell_cmd)
        self.assertIn("CODEX_WEB_LAUNCH_ID=", shell_cmd)
        self.assertIn("unset CODEX_HOME PI_HOME CLAUDE_CONFIG_DIR CODEX_BIN PI_BIN CLAUDE_BIN CODEX_WEB_OWNER", shell_cmd)
        self.assertIn("CODEX_WEB_RESUME_SESSION_ID", shell_cmd)
        self.assertNotIn("CODEX_WEB_RESUME_SESSION_ID=", shell_cmd)
        self.assertIn("CODEX_WEB_MODEL_PROVIDER=crs", shell_cmd)
        self.assertIn("CODEX_WEB_PREFERRED_AUTH_METHOD=apikey", shell_cmd)
        self.assertIn("CODEX_WEB_MODEL=gpt-5.4", shell_cmd)
        self.assertIn("CODEX_WEB_SERVICE_TIER=fast", shell_cmd)
        self.assertIn("check_for_update_on_startup=false", shell_cmd)
        self.assertIn("--disable", shell_cmd)
        self.assertIn("goals", shell_cmd)
        self.assertIn("codoxear.broker", shell_cmd)

    def test_tmux_metadata_delay_returns_pending_with_real_snapshot_shape(self) -> None:
        coordinator = _web_launch_coordinator(
            which_tmux=lambda _name: "/usr/bin/tmux",
            wait_for_spawned_broker_meta=lambda _nonce: (_ for _ in ()).throw(TimeoutError("metadata not ready")),
            tmux_pane_snapshot=lambda *a, **kw: {"tmux_pane_id": "%8", "tmux_pane_dead": "0", "tmux_window": "work-abc123"},
            run=lambda *a, **kw: subprocess.CompletedProcess(["/usr/bin/tmux", "new-window"], 0, stdout="%8\n", stderr=""),
        )
        with TemporaryDirectory() as td:
            result = coordinator.spawn_web_session(cwd=td, create_in_tmux=True)

        self.assertEqual(result["pending"], True)
        self.assertEqual(result["tmux_session"], "codoxear")
        self.assertIsInstance(result["launch_id"], str)

    def test_spawn_web_session_rejects_tmux_when_unavailable(self) -> None:
        coordinator = _web_launch_coordinator(which_tmux=lambda _name: None)
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "tmux is unavailable on this host"):
                coordinator.spawn_web_session(cwd=td, create_in_tmux=True)

    def test_spawn_web_session_rejects_resume_id_not_in_cwd(self) -> None:
        coordinator = _web_launch_coordinator(
            list_resume_candidates_for_cwd=lambda cwd, *, agent_backend, limit: [],
        )
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "resume session not found for cwd"):
                coordinator.spawn_web_session(cwd=td, resume_session_id="missing")

    def test_spawn_web_session_passes_pi_backend_to_broker(self) -> None:
        thread_starts, start_fn = _capture_thread_starts()
        popen = _PopenRecorder(_FakeProc(7654))
        coordinator = _web_launch_coordinator(
            wait_or_raise=lambda *a, **kw: None,
            popen=popen,
        )
        with TemporaryDirectory() as td, patch.object(threading.Thread, "start", start_fn):
            result = coordinator.spawn_web_session(
                cwd=td,
                agent_backend="pi",
                model_provider="macaron",
                model="gpt-5.4",
                reasoning_effort="medium",
            )

        self.assertEqual(
            popen.argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                td,
                "--",
                "--provider",
                "macaron",
                "--model",
                "gpt-5.4",
                "--thinking",
                "medium",
            ],
        )
        self.assertEqual(popen.env["CODEX_WEB_AGENT_BACKEND"], "pi")
        self.assertEqual(popen.env["PI_HOME"], str(Path.home() / ".pi"))
        self.assertEqual(popen.env["CODEX_WEB_MODEL_PROVIDER"], "macaron")
        self.assertEqual(popen.env["CODEX_WEB_MODEL"], "gpt-5.4")
        self.assertEqual(popen.env["CODEX_WEB_REASONING_EFFORT"], "medium")
        self.assertNotIn("--disable", popen.argv)
        self.assertNotIn("goals", popen.argv)
        self.assertEqual(result, {"broker_pid": 7654})
        self.assertEqual(thread_starts, ["start"])

    def test_spawn_web_session_passes_cc_backend_to_broker(self) -> None:
        thread_starts, start_fn = _capture_thread_starts()
        popen = _PopenRecorder(_FakeProc(8765))
        coordinator = _web_launch_coordinator(
            wait_or_raise=lambda *a, **kw: None,
            popen=popen,
        )
        with TemporaryDirectory() as td, patch.object(threading.Thread, "start", start_fn):
            result = coordinator.spawn_web_session(
                cwd=td,
                agent_backend="cc",
                model="claude-haiku-4-5",
                reasoning_effort="max",
            )

        self.assertEqual(
            popen.argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                td,
                "--",
                "--dangerously-skip-permissions",
                "--model",
                "claude-haiku-4-5",
                "--effort",
                "max",
            ],
        )
        self.assertEqual(popen.env["CODEX_WEB_AGENT_BACKEND"], "cc")
        self.assertEqual(popen.env["CLAUDE_CONFIG_DIR"], str(Path.home() / ".claude"))
        self.assertEqual(popen.env["CODEX_WEB_MODEL"], "claude-haiku-4-5")
        self.assertEqual(popen.env["CODEX_WEB_REASONING_EFFORT"], "max")
        self.assertNotIn("CODEX_HOME", popen.env)
        self.assertNotIn("PI_HOME", popen.env)
        self.assertEqual(result, {"broker_pid": 8765})
        self.assertEqual(thread_starts, ["start"])

    def test_spawn_web_session_passes_cc_resume_id(self) -> None:
        thread_starts, start_fn = _capture_thread_starts()
        popen = _PopenRecorder(_FakeProc(8766))
        coordinator = _web_launch_coordinator(
            list_resume_candidates_for_cwd=lambda cwd, *, agent_backend, limit: [{"session_id": "resume-cc"}],
            wait_or_raise=lambda *a, **kw: None,
            popen=popen,
        )
        with TemporaryDirectory() as td, patch.object(threading.Thread, "start", start_fn):
            result = coordinator.spawn_web_session(
                cwd=td,
                agent_backend="cc",
                resume_session_id="resume-cc",
            )

        self.assertEqual(
            popen.argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                td,
                "--",
                "--dangerously-skip-permissions",
                "--resume",
                "resume-cc",
            ],
        )
        self.assertEqual(result, {"broker_pid": 8766})
        self.assertEqual(thread_starts, ["start"])

    def test_spawn_web_session_uses_pi_session_arg_without_resume_log_env(self) -> None:
        thread_starts, start_fn = _capture_thread_starts()
        popen = _PopenRecorder(_FakeProc(7655))
        coordinator = _web_launch_coordinator(
            list_resume_candidates_for_cwd=lambda cwd, *, agent_backend, limit: [
                {"session_id": "resume-a", "log_path": "/tmp/pi-resume.jsonl"}
            ],
            wait_or_raise=lambda *a, **kw: None,
            popen=popen,
        )
        with TemporaryDirectory() as td, patch.object(threading.Thread, "start", start_fn):
            result = coordinator.spawn_web_session(
                cwd=td,
                agent_backend="pi",
                resume_session_id="resume-a",
            )

        self.assertEqual(
            popen.argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                td,
                "--",
                "--session",
                "/tmp/pi-resume.jsonl",
            ],
        )
        self.assertEqual(popen.env["CODEX_WEB_RESUME_SESSION_ID"], "resume-a")
        self.assertNotIn("CODEX_WEB_RESUME_LOG_PATH", popen.env)
        self.assertEqual(result, {"broker_pid": 7655})
        self.assertEqual(thread_starts, ["start"])

    def test_create_git_worktree_creates_new_checkout(self) -> None:
        # Real-git integration test; no monkeypatch seams. Exercises the shared
        # git_ops helpers directly (the server facades are thin pass-throughs).
        with TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (root / "README.md").write_text("x\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            branch = "feature/test-worktree"
            worktree = git_ops.default_worktree_path(root, branch)
            result = git_ops.create_git_worktree(
                root, branch, git_repo_root_func=server._git_repo_root, timeout_s=server.GIT_WORKTREE_TIMEOUT_SECONDS
            )
            self.assertEqual(result, worktree.resolve())
            self.assertTrue((worktree / ".git").exists())
            branch_name = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=worktree,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            self.assertEqual(branch_name, branch)

    def test_spawn_web_session_uses_created_worktree_as_cwd(self) -> None:
        thread_starts, start_fn = _capture_thread_starts()
        popen = _PopenRecorder(_FakeProc(5432))
        with TemporaryDirectory() as td:
            coordinator = _web_launch_coordinator(
                create_git_worktree=lambda _source, _branch: Path(td) / "repo-worktree",
                wait_or_raise=lambda *a, **kw: None,
                popen=popen,
            )
            with patch.object(threading.Thread, "start", start_fn):
                result = coordinator.spawn_web_session(cwd=td, worktree_branch="feature/test-worktree")

        trust_override = f'projects={{ {json.dumps(str((Path(td) / "repo-worktree").resolve()))} = {{ trust_level = "trusted" }} }}'
        self.assertEqual(
            popen.argv,
            [
                ANY,
                "-m",
                "codoxear.broker",
                "--cwd",
                str(Path(td) / "repo-worktree"),
                "--",
                "-c",
                trust_override,
                "-c",
                "check_for_update_on_startup=false",
                "--disable",
                "goals",
                "--dangerously-bypass-approvals-and-sandbox",
            ],
        )
        self.assertEqual(result, {"broker_pid": 5432})
        self.assertEqual(thread_starts, ["start"])

    def test_spawn_web_session_rejects_worktree_when_resuming(self) -> None:
        coordinator = _web_launch_coordinator()
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "worktree_branch cannot be used when resuming a session"):
                coordinator.spawn_web_session(cwd=td, resume_session_id="resume-a", worktree_branch="feature/test-worktree")

    def test_spawn_web_session_rejects_worktree_outside_git(self) -> None:
        # Real create_git_worktree raises when cwd is not inside a git repo; the
        # temp dir is not a git repo, so no patch is needed.
        coordinator = _web_launch_coordinator()
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "cwd is not inside a git worktree"):
                coordinator.spawn_web_session(cwd=td, worktree_branch="feature/test-worktree")

    def test_resume_catchup_suppresses_delivery_until_resume_marker_clears(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_path = root / "broker.sock"
            sock_path.touch()
            log_path = root / "rollout-2026-03-29T10-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            meta_path = sock_path.with_suffix(".json")

            replay_row = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "historical replay reply"}],
                },
                "ts": 1.0,
            }
            _write_jsonl(
                log_path,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "thread-1",
                            "cwd": str(root),
                            "source": "cli",
                        },
                    },
                    replay_row,
                ],
            )
            meta_path.write_text(
                json.dumps(
                    {
                        "session_id": "thread-1",
                        "owner": "web",
                        "broker_pid": 1,
                        "codex_pid": 2,
                        "cwd": str(root),
                        "start_ts": 100.0,
                        "log_path": str(log_path),
                        "sock_path": str(sock_path),
                        "resume_session_id": "resume-old",
                    }
                ),
                encoding="utf-8",
            )

            voice_push = _FakeVoicePush()
            session = Session(
                session_id="broker-1",
                thread_id="thread-1",
                broker_pid=1,
                codex_pid=2,
                agent_backend="codex",
                owned=True,
                start_ts=100.0,
                cwd=str(root),
                log_path=log_path,
                sock_path=sock_path,
            )
            sessions = {session.session_id: session}
            refresh = _refresh_coordinator(
                sessions=sessions,
                proc_root=root,
                proc_find_open_rollout_log=lambda **_kw: None,
                sock_call=lambda *a, **kw: {"tail": ""},
            )
            voice = _voice_coordinator(sessions=sessions, voice_push=voice_push)

            refresh.refresh_session_meta(session.session_id)
            voice.observe_rollout_delta(session.session_id, objs=[replay_row], new_off=10)
            self.assertEqual(voice_push.calls, [])
            self.assertEqual(session.delivery_log_off, 10)
            self.assertEqual(session.resume_session_id, "resume-old")

            fresh_row = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "fresh reply after resume"}],
                },
                "ts": 2.0,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(fresh_row) + "\n")
            meta_path.write_text(
                json.dumps(
                    {
                        "session_id": "thread-1",
                        "owner": "web",
                        "broker_pid": 1,
                        "codex_pid": 2,
                        "cwd": str(root),
                        "start_ts": 100.0,
                        "log_path": str(log_path),
                        "sock_path": str(sock_path),
                        "resume_session_id": None,
                    }
                ),
                encoding="utf-8",
            )

            refresh.refresh_session_meta(session.session_id)
            voice.observe_rollout_delta(session.session_id, objs=[fresh_row], new_off=20)
            self.assertIsNone(session.resume_session_id)
            self.assertEqual(len(voice_push.calls), 1)
            delivered = voice_push.calls[0]
            self.assertEqual(delivered["session_id"], "broker-1")
            messages = delivered["messages"]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].text, "fresh reply after resume")

    def test_claude_split_tool_delta_delivery_keeps_final_as_narration(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "claude.jsonl"
            tool_row = {
                "type": "assistant",
                "sessionId": "cc-thread",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}],
                    "stop_reason": "tool_use",
                },
            }
            final_row = {
                "type": "assistant",
                "sessionId": "cc-thread",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
            }
            first = json.dumps(tool_row) + "\n"
            log_path.write_text(first + json.dumps(final_row) + "\n", encoding="utf-8")

            voice_push = _FakeVoicePush()
            session = Session(
                session_id="broker-cc",
                thread_id="cc-thread",
                broker_pid=1,
                codex_pid=2,
                agent_backend="cc",
                owned=True,
                start_ts=100.0,
                cwd=str(root),
                log_path=log_path,
                sock_path=root / "broker.sock",
            )
            voice = _voice_coordinator(sessions={session.session_id: session}, voice_push=voice_push)

            voice.observe_rollout_delta(
                session.session_id, log_path=log_path, old_off=len(first), objs=[final_row], new_off=log_path.stat().st_size
            )
            self.assertEqual(len(voice_push.calls), 1)
            messages = voice_push.calls[0]["messages"]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].message_class, "narration")

    def test_discover_existing_binds_open_rollout_when_sidecar_has_no_log(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            sock_path = sock_dir / "broker-1.sock"
            sock_path.touch()
            log_path = root / "rollout-2026-05-13T10-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            _write_jsonl(
                log_path,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "thread-1",
                            "cwd": str(root),
                            "source": "cli",
                        },
                    }
                ],
            )
            sock_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": None,
                        "owner": "web",
                        "broker_pid": 11,
                        "codex_pid": 12,
                        "cwd": str(root),
                        "start_ts": 100.0,
                        "log_path": None,
                        "sock_path": str(sock_path),
                    }
                ),
                encoding="utf-8",
            )

            deps = _discovery_deps(proc_find_open_rollout_log=lambda *a, **kw: log_path)
            result = discover_sessions(sock_dir, proc_root=root, hidden_sessions=set(), deps=deps)
            self.assertEqual(len(result.registrations), 1)
            reg = result.registrations[0]
            self.assertEqual(reg.thread_id, "thread-1")
            self.assertEqual(reg.log_path, log_path)

            # Strengthen: a subsequent refresh against the discovered session
            # keeps the log bound (proc_find_open_rollout_log returns it again).
            session = _registration_to_session(reg)
            sessions = {session.session_id: session}
            refresh = _refresh_coordinator(
                sessions=sessions,
                proc_root=root,
                proc_find_open_rollout_log=lambda *a, **kw: log_path,
                sock_call=lambda *a, **kw: {"tail": ""},
            )
            refresh.refresh_session_meta(session.session_id)

            self.assertEqual(session.thread_id, "thread-1")
            self.assertEqual(session.log_path, log_path)

    def test_discover_existing_does_not_rebind_ignored_detached_rollout(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            sock_path = sock_dir / "broker-1.sock"
            sock_path.touch()
            old_log = root / "rollout-2026-05-13T10-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            _write_jsonl(
                old_log,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "old-thread",
                            "cwd": str(root),
                            "source": "cli",
                        },
                    }
                ],
            )
            sock_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": None,
                        "owner": "web",
                        "broker_pid": 11,
                        "codex_pid": 12,
                        "cwd": str(root),
                        "start_ts": 100.0,
                        "log_path": None,
                        "ignored_rollout_paths": [str(old_log)],
                        "sock_path": str(sock_path),
                    }
                ),
                encoding="utf-8",
            )

            def _find_open(*args, **kwargs):
                ignored = kwargs.get("ignored_paths")
                if ignored is None and len(args) >= 5:
                    ignored = args[4]
                self.assertEqual(ignored, {old_log})
                return None

            deps = _discovery_deps(proc_find_open_rollout_log=_find_open)
            result = discover_sessions(sock_dir, proc_root=root, hidden_sessions=set(), deps=deps)
            self.assertEqual(len(result.registrations), 1)
            reg = result.registrations[0]
            self.assertEqual(reg.thread_id, "broker-1")
            self.assertIsNone(reg.log_path)

            session = _registration_to_session(reg)
            sessions = {session.session_id: session}
            refresh = _refresh_coordinator(
                sessions=sessions,
                proc_root=root,
                proc_find_open_rollout_log=_find_open,
                sock_call=lambda *a, **kw: {"tail": ""},
            )
            refresh.refresh_session_meta(session.session_id)

            self.assertEqual(session.thread_id, "broker-1")
            self.assertIsNone(session.log_path)

    def test_refresh_session_meta_treats_null_sidecar_as_detach_for_existing_log(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            sock_path = sock_dir / "broker-1.sock"
            sock_path.touch()
            old_log = root / "rollout-2026-05-13T10-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            _write_jsonl(
                old_log,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "old-thread",
                            "cwd": str(root),
                            "source": "cli",
                        },
                    }
                ],
            )
            sock_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": None,
                        "owner": "web",
                        "broker_pid": 11,
                        "codex_pid": 12,
                        "cwd": str(root),
                        "start_ts": 100.0,
                        "log_path": None,
                        "sock_path": str(sock_path),
                    }
                ),
                encoding="utf-8",
            )

            session = Session(
                session_id="broker-1",
                thread_id="old-thread",
                broker_pid=11,
                codex_pid=12,
                agent_backend="codex",
                owned=True,
                start_ts=100.0,
                cwd=str(root),
                log_path=old_log,
                sock_path=sock_path,
            )

            def _sock_call(_sock: Path, req: dict[str, object], **_kwargs: object) -> dict[str, object]:
                if req.get("cmd") == "tail":
                    return {"tail": "To continue this session, run codex resume old-thread"}
                return {"busy": False, "queue_len": 0, "token": None}

            def _find_open(*args, **kwargs) -> Path | None:
                ignored = kwargs.get("ignored_paths")
                if ignored is None and len(args) >= 5:
                    ignored = args[4]
                self.assertEqual(ignored, {old_log})
                return None

            refresh = _refresh_coordinator(
                sessions={session.session_id: session},
                proc_root=root,
                proc_find_open_rollout_log=_find_open,
                sock_call=_sock_call,
            )
            refresh.refresh_session_meta(session.session_id)

            self.assertEqual(session.thread_id, "old-thread")
            self.assertIsNone(session.log_path)

    def test_refresh_session_meta_still_repairs_stale_null_sidecar_without_detach_marker(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            sock_dir = root / "socks"
            sock_dir.mkdir()
            sock_path = sock_dir / "broker-1.sock"
            sock_path.touch()
            log_path = root / "rollout-2026-05-13T10-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            _write_jsonl(
                log_path,
                [
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "thread-1",
                            "cwd": str(root),
                            "source": "cli",
                        },
                    }
                ],
            )
            sock_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "session_id": None,
                        "owner": "web",
                        "broker_pid": 11,
                        "codex_pid": 12,
                        "cwd": str(root),
                        "start_ts": 100.0,
                        "log_path": None,
                        "sock_path": str(sock_path),
                    }
                ),
                encoding="utf-8",
            )

            session = Session(
                session_id="broker-1",
                thread_id="broker-1",
                broker_pid=11,
                codex_pid=12,
                agent_backend="codex",
                owned=True,
                start_ts=100.0,
                cwd=str(root),
                log_path=log_path,
                sock_path=sock_path,
            )

            def _sock_call(_sock: Path, req: dict[str, object], **_kwargs: object) -> dict[str, object]:
                if req.get("cmd") == "tail":
                    return {"tail": "plain startup output"}
                return {"busy": False, "queue_len": 0, "token": None}

            refresh = _refresh_coordinator(
                sessions={session.session_id: session},
                proc_root=root,
                proc_find_open_rollout_log=lambda *a, **kw: log_path,
                sock_call=_sock_call,
            )
            refresh.refresh_session_meta(session.session_id)

            self.assertEqual(session.thread_id, "thread-1")
            self.assertEqual(session.log_path, log_path)


if __name__ == "__main__":
    unittest.main()
