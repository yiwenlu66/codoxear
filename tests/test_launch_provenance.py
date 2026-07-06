"""Direct impl/coordinator tests for launch-provenance / failure-detail logic.

Previously these tests patched the ``codoxear.server`` module surface (~9 sites):
``server.LAUNCH_ATTEMPTS_PATH`` (7 tests), ``server.sys.stderr`` and
``codoxear.server.shutil.which`` / ``codoxear.server._tmux_pane_snapshot``.
They reached the launch ledger only through the thin ``server._*`` wrappers that
bind module globals (``LAUNCH_ATTEMPTS_PATH``, ``sys.stderr``,
``DEFAULT_AGENT_BACKEND`` ...).

They now exercise the true seams directly with injected dependencies:

* append/read/collapse of the launch ledger -> ``codoxear.util.append_launch_attempt``
  / ``read_launch_attempts`` and ``codoxear.launch_ledger.record_launch_attempt``
  called with a real temp ``path`` (and an injected ``stderr``). The file-system
  path replaces the patched ``server.LAUNCH_ATTEMPTS_PATH`` module global.
* transcript / row / redaction helpers -> the pure
  ``codoxear.launch_ledger`` (``launch_attempt_transcript_payload``,
  ``launch_attempt_row``) and ``codoxear.util.redact_launch_failure_text``
  free functions directly, bound to the same ``DEFAULT_AGENT_BACKEND`` /
  unattended constants the server wrapper binds. No facade indirection.
* ``SessionLaunchError`` response shaping -> ``codoxear.session_errors``
  directly instead of the re-exported ``server.SessionLaunchError`` alias.
* ``list_sessions`` exposing recent failed / pending launch rows, hiding rows
  for active launches, and omitting successful launches -> the real
  ``SessionListCoordinator.list_sessions`` wired to a real ``SessionStore``;
  ``read_launch_attempts`` reads the temp ledger, ``launch_attempt_row`` is the
  impl, ``include_launch_attempts`` is injected.
* dismiss-on-delete and kill-on-delete launch hiding ->
  ``SessionLifecycleCoordinator.delete_session`` sharing the same store and
  hidden-session set as the listing coordinator, with ``sock_call`` injected as
  the socket boundary.
* dead-session pruning that preserves an existing launch failure vs. records a
  new pre-log failure carrying submitted user messages and the tmux pane tail ->
  ``SessionPruneCoordinator.prune_dead_sessions`` with ``latest_launch_attempt``
  / ``record_launch_attempt`` bound to the temp ledger and the
  ``which_tmux`` / ``tmux_pane_snapshot`` OS/process boundaries injected as
  coordinator fields.

No ``codoxear.server.*`` module-global monkeypatching remains; no ``patch(...)``
call is retained at all. No file under ``codoxear/`` is modified. No
``try/except`` swallows.
"""

import io
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear import server
from codoxear.launch_ledger import launch_attempt_row
from codoxear.launch_ledger import launch_attempt_transcript_payload
from codoxear.launch_ledger import launch_failure_tail
from codoxear.launch_ledger import latest_launch_attempt
from codoxear.launch_ledger import record_launch_attempt
from codoxear.launch_ledger import submitted_user_messages
from codoxear.post_log_recovery import POST_LOG_BOUND_BACKEND_STOPPED_TEXT
from codoxear.post_log_recovery import log_needs_post_log_bound_recovery
from codoxear.session_errors import SessionLaunchError
from codoxear.session_lifecycle import SessionLifecycleCoordinator
from codoxear.session_list import SessionListCoordinator
from codoxear.session_model import Session
from codoxear.session_prune import SessionPruneCoordinator
from codoxear.session_runtime import ListingRuntimeProbes
from codoxear.session_runtime import log_path_size_or_none
from codoxear.session_store import SessionStore
from codoxear.session_store import SessionStorePaths
from codoxear.util import append_launch_attempt
from codoxear.util import read_launch_attempts
from codoxear.util import redact_launch_failure_text

DEFAULT_AGENT_BACKEND = server.DEFAULT_AGENT_BACKEND
UNATTENDED_DEFAULT_IDLE_MINUTES = server.UNATTENDED_DEFAULT_IDLE_MINUTES
UNATTENDED_DEFAULT_MAX_INJECTIONS = server.UNATTENDED_DEFAULT_MAX_INJECTIONS


# --------------------------------------------------------------------------- #
# Impl bindings — same constants the ``server._*`` wrappers bind, called
# directly so no module global is patched.
# --------------------------------------------------------------------------- #


def _launch_attempt_row_from_record(record: dict) -> dict | None:
    """Mirror of ``server._launch_attempt_row``: bind the launch-row impl to the
    server's default-agent-backend / unattended constants. Calling it directly
    removes the need to patch ``server.LAUNCH_ATTEMPTS_PATH`` to exercise the
    row projection used by ``list_sessions``."""
    return launch_attempt_row(
        record,
        default_agent_backend=DEFAULT_AGENT_BACKEND,
        unattended_default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
    )


# --------------------------------------------------------------------------- #
# Coordinator / store builders (real coordinators wired to real stores; tests
# mutate the store dicts and the coordinators observe the same state).
# --------------------------------------------------------------------------- #


def _store(root: Path) -> SessionStore:
    """Real ``SessionStore`` rooted at a temp dir, wired to the server's real
    cleaner functions. Tests mutate its in-memory dicts directly; the listing /
    lifecycle / prune coordinators read and mutate the same store."""
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
        unattended_default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
        clean_alias=server._clean_alias,
        clean_priority_offset=server._clean_priority_offset,
        clean_snooze_until=server._clean_snooze_until,
        clean_dependency_session_id=server._clean_dependency_session_id,
        clean_recent_cwd=server._clean_recent_cwd,
        clean_commit_unknown_send_record=lambda value: value if isinstance(value, dict) else None,
    )


def _probes() -> ListingRuntimeProbes:
    """``ListingRuntimeProbes`` wired to the real server readers. The launch-row
    tests have no live sessions, so these probes are not exercised; they are
    provided so the listing coordinator is fully and truthfully constructed."""
    return ListingRuntimeProbes(
        last_conversation_ts_from_tail=server._last_conversation_ts_from_tail,
        read_run_settings_from_log=lambda path, agent_backend: server._read_run_settings_from_log(
            path, agent_backend=agent_backend
        ),
        log_size_or_none=log_path_size_or_none,
        send_boundary_unresolved=lambda _sid, _path, _size: False,
        idle_from_log_path=lambda _sid, _path: True,
        current_git_branch=server._current_git_branch,
    )


def _list_coordinator(
    *,
    store: SessionStore,
    sessions: dict[str, Session],
    lock: threading.Lock,
    ledger_path: Path,
    include_launch_attempts: bool = True,
) -> SessionListCoordinator:
    """``SessionListCoordinator`` wired to the real store and the real server
    cleaners/providers. ``read_launch_attempts`` reads the temp ledger and
    ``launch_attempt_row`` is the bound impl, so launch-row projection is
    exercised through the true ``list_sessions`` path without patching
    ``server.LAUNCH_ATTEMPTS_PATH``."""
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
        save_files=lambda: None,
        save_sidebar_meta=lambda: None,
        save_recent_cwds=lambda: None,
        now=time.time,
        runtime_probes=_probes(),
        include_launch_attempts=lambda: include_launch_attempts,
        read_launch_attempts=lambda: read_launch_attempts(path=ledger_path, max_records=100, max_age_s=24 * 3600),
        launch_attempt_row=_launch_attempt_row_from_record,
        clean_unattended_cooldown_minutes=server._clean_unattended_cooldown_minutes,
        clean_unattended_remaining_injections=server._clean_unattended_remaining_injections,
        provider_choice_for_settings=server._provider_choice_for_settings,
        resolve_session_cwd=server._resolve_session_cwd,
        unattended_default_idle_minutes=UNATTENDED_DEFAULT_IDLE_MINUTES,
        unattended_default_max_injections=UNATTENDED_DEFAULT_MAX_INJECTIONS,
        priority_half_life_seconds=server.SIDEBAR_PRIORITY_HALF_LIFE_SECONDS,
        priority_bucket_seconds=server.SIDEBAR_PRIORITY_BUCKET_SECONDS,
    )


def _lifecycle_coordinator(
    *,
    store: SessionStore,
    sessions: dict[str, Session],
    lock: threading.Lock,
    ledger_path: Path,
    sock_call,
) -> SessionLifecycleCoordinator:
    """``SessionLifecycleCoordinator`` sharing the listing coordinator's store
    and hidden-session set. ``sock_call`` is the injected socket boundary; the
    launch ledger is read directly from the temp path via the bound row impl."""
    return SessionLifecycleCoordinator(
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
        read_launch_attempts=lambda: read_launch_attempts(path=ledger_path, max_records=100, max_age_s=24 * 3600),
        launch_attempt_row=_launch_attempt_row_from_record,
        hide_session=lambda sid: store.hidden_sessions.add(sid),
    )


class TestLaunchProvenance(unittest.TestCase):
    def test_launch_attempt_log_collapses_to_latest_state(self) -> None:
        # Already a direct impl test: append/read the real temp ledger via the
        # util free functions (path is the injected dependency). No server
        # module global is touched.
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
        # SessionListCoordinator.list_sessions reads the temp ledger directly
        # (read_launch_attempts closure) and projects rows through the bound
        # launch_attempt_row impl, replacing the patched server.LAUNCH_ATTEMPTS_PATH.
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "launches.jsonl"
            store = _store(root)
            lock = threading.Lock()
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
                path=ledger,
            )

            coordinator = _list_coordinator(store=store, sessions={}, lock=lock, ledger_path=ledger)
            rows = coordinator.list_sessions()

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

    def test_failed_launch_transcript_exposes_submitted_user_message(self) -> None:
        # launch_attempt_transcript_payload is a pure free function in
        # launch_ledger; call it directly instead of through server._launch_attempt_transcript_payload.
        now = time.time()
        rec = {
            "launch_id": "launch-dead",
            "state": "failed",
            "stage": "agent_exit_before_log_bind",
            "error": "codex exited with status 1 before a session log was bound",
            "agent_backend": "codex",
            "cwd": "/tmp/work",
            "created_ts": now,
            "updated_ts": now + 1.0,
            "agent_exit_status": 1,
            "broker_exit_status": 1,
            "pty_tail": "fatal: provider unavailable\n",
            "submitted_user_messages": [{"text": "Please recover this prompt", "ts": now + 0.5, "source": "send"}],
        }

        payload = launch_attempt_transcript_payload(rec)

        self.assertEqual(payload["transcript_state"], "failed")
        self.assertEqual(payload["thread_id"], "launch-dead")
        self.assertEqual(payload["events"][0]["role"], "user")
        self.assertEqual(payload["events"][0]["text"], "Please recover this prompt")
        self.assertEqual(payload["events"][1]["role"], "assistant")
        self.assertEqual(payload["events"][1]["message_class"], "error")
        self.assertIn("Agent exit status: 1", payload["events"][1]["text"])
        self.assertIn("fatal: provider unavailable", payload["events"][1]["text"])

    def test_post_log_bound_recovery_row_uses_routing_session_id_and_preserves_thread(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "post-log.jsonl"
            log_path.write_text(
                '{"type":"event_msg","ts":1.0,"payload":{"type":"user_message","message":"POST_LOG_BOUND_DEATH_SENTINEL"}}\n',
                encoding="utf-8",
            )
            rec = {
                "launch_id": "launch-post-log",
                "session_id": "broker-123",
                "thread_id": "rollout-thread-abc",
                "state": "failed",
                "stage": "session_pruned_after_log_bind",
                "error": "backend stopped after binding a transcript log",
                "agent_backend": "codex",
                "cwd": str(root),
                "created_ts": time.time(),
                "updated_ts": time.time() + 1.0,
                "broker_pid": 123,
                "agent_pid": 124,
                "log_path": str(log_path),
            }

            row = _launch_attempt_row_from_record(rec)
            payload = launch_attempt_transcript_payload(rec)

        assert row is not None
        self.assertEqual(row["session_id"], "broker-123")
        self.assertEqual(row["thread_id"], "rollout-thread-abc")
        self.assertEqual(row["launch_id"], "launch-post-log")
        self.assertEqual(row["log_path"], str(log_path))
        self.assertEqual(row["busy"], False)
        self.assertEqual(payload["session_id"], "broker-123")
        self.assertEqual(payload["thread_id"], "rollout-thread-abc")
        self.assertEqual([ev["role"] for ev in payload["events"]], ["user", "assistant"])
        self.assertEqual(payload["events"][0]["text"], "POST_LOG_BOUND_DEATH_SENTINEL")
        self.assertEqual(payload["events"][1]["message_class"], "error")
        self.assertIn("stopped before completing", payload["events"][1]["text"])

    def test_pre_log_failed_launch_row_still_uses_launch_id(self) -> None:
        rec = {
            "launch_id": "launch-pre-log",
            "session_id": "broker-should-not-route",
            "state": "failed",
            "stage": "agent_exit_before_log_bind",
            "error": "agent exited before log bind",
            "agent_backend": "codex",
            "cwd": "/tmp/work",
            "created_ts": time.time(),
        }

        row = _launch_attempt_row_from_record(rec)

        assert row is not None
        self.assertEqual(row["session_id"], "launch-pre-log")
        self.assertEqual(row["thread_id"], "launch-pre-log")
        self.assertIsNone(row["log_path"])

    def test_post_log_recovery_needed_only_for_non_idle_bound_logs(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            incomplete = root / "incomplete.jsonl"
            incomplete.write_text(
                '{"type":"event_msg","ts":1.0,"payload":{"type":"user_message","message":"unfinished"}}\n',
                encoding="utf-8",
            )
            complete = root / "complete.jsonl"
            complete.write_text(
                "".join(
                    [
                        '{"type":"event_msg","ts":1.0,"payload":{"type":"user_message","message":"done"}}\n',
                        '{"type":"event_msg","ts":2.0,"payload":{"type":"agent_message","message":"answer","phase":"final_answer"}}\n',
                        '{"type":"event_msg","ts":3.0,"payload":{"type":"task_complete","turn_id":"done","last_agent_message":"answer"}}\n',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(log_needs_post_log_bound_recovery(incomplete))
            self.assertFalse(log_needs_post_log_bound_recovery(complete))

    def test_broker_post_log_recovery_source_uses_socket_route_identity(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "codoxear" / "broker.py").read_text(encoding="utf-8")

        self.assertIn("session_id=st2.sock_path.stem if st2.sock_path else st2.session_id", source)
        self.assertIn("thread_id=st2.session_id", source)

    def test_failed_launch_transcript_strips_ansi_from_terminal_tail(self) -> None:
        now = time.time()
        rec = {
            "launch_id": "launch-ansi",
            "state": "failed",
            "stage": "agent_exit_before_log_bind",
            "error": "claude exited with status 1 before a session log was bound",
            "agent_backend": "cc",
            "cwd": "/tmp/work",
            "created_ts": now,
            "updated_ts": now,
            "pty_tail": "File \x1b[35m\"<string>\"\x1b[0m, line \x1b[35m11\x1b[0m\n\x1b[1;35mFileNotFoundError\x1b[0m: \x1b[35mNo such file or directory: b'/usr/games/claude'\x1b[0m\n",
        }

        payload = launch_attempt_transcript_payload(rec)
        text = payload["events"][0]["text"]

        self.assertIn('File "<string>", line 11', text)
        self.assertIn("FileNotFoundError: No such file or directory: b'/usr/games/claude'", text)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("[35m", text)
        self.assertNotIn("[0m", text)

    def test_failed_launch_transcript_redacts_error_and_tail_secrets(self) -> None:
        now = time.time()
        rec = {
            "launch_id": "launch-secret",
            "state": "failed",
            "stage": "shell_startup",
            "error": "failed API_TOKEN: secret-token password: hunter2 \"api_key\":\"json-secret\" Bearer abcdefghijklmnop",
            "agent_backend": "codex",
            "cwd": "/tmp/work",
            "created_ts": now,
            "updated_ts": now,
            "pty_tail": "tail OPENAI_API_KEY: tail-secret sk-abcdefghijklmnop\n",
        }

        payload = launch_attempt_transcript_payload(rec)
        text = payload["events"][0]["text"]

        self.assertIn("API_TOKEN: [redacted]", text)
        self.assertIn("password: [redacted]", text)
        self.assertIn('"api_key":[redacted]', text)
        self.assertIn("OPENAI_API_KEY: [redacted]", text)
        self.assertNotIn("secret-token", text)
        self.assertNotIn("hunter2", text)
        self.assertNotIn("json-secret", text)
        self.assertNotIn("tail-secret", text)
        self.assertNotIn("abcdefghijklmnop", text)

    def test_session_launch_error_exposes_only_redacted_record(self) -> None:
        # SessionLaunchError lives in codoxear.session_errors; construct it
        # directly instead of via the server re-export.
        err = SessionLaunchError(
            {
                "launch_id": "launch-secret-response",
                "state": "failed",
                "error": "failed API_TOKEN: top-secret password: hunter2",
                "tmux_stderr": "stderr API_TOKEN=stderr-secret",
                "tmux_attempts": [{"stderr": "nested PASSWORD=\"nested-secret"}],
                "metadata": {"error": "OPENAI_API_KEY=meta-secret"},
            }
        )

        self.assertEqual(str(err), "failed API_TOKEN: [redacted] password: [redacted]")
        self.assertEqual(err.record["error"], "failed API_TOKEN: [redacted] password: [redacted]")
        self.assertEqual(err.record["launch_id"], "launch-secret-response")
        self.assertEqual(err.record["state"], "failed")
        self.assertNotIn("tmux_stderr", err.record)
        self.assertNotIn("tmux_attempts", err.record)
        self.assertNotIn("metadata", err.record)
        self.assertNotIn("top-secret", str(err.record))
        self.assertNotIn("hunter2", str(err.record))
        self.assertNotIn("stderr-secret", str(err.record))
        self.assertNotIn("nested-secret", str(err.record))
        self.assertNotIn("meta-secret", str(err.record))

    def test_record_launch_attempt_redacts_persisted_record_and_stderr(self) -> None:
        # record_launch_attempt takes path and stderr as injected kwargs; the
        # temp path replaces server.LAUNCH_ATTEMPTS_PATH and the StringIO
        # replaces server.sys.stderr (no module-global patching).
        with TemporaryDirectory() as td:
            path = Path(td) / "launches.jsonl"
            stderr = io.StringIO()
            rec = record_launch_attempt(
                {
                    "launch_id": "launch-persist-secret",
                    "state": "failed",
                    "stage": "tmux_start",
                    "error": "top API_TOKEN: top-secret \"api_key\":\"json-secret\" Authorization: Bearer auth-secret-token",
                    "tmux_stderr": "stderr API_TOKEN: stderr-secret AUTH: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==",
                    "tmux_attempts": [{"stderr": "nested password: nested-secret"}],
                    "metadata": {"api_key": "meta-secret", "auth_header": "custom-secret"},
                },
                path=path,
                stderr=stderr,
            )

            rows = read_launch_attempts(path=path, max_records=10, max_age_s=3600)
            persisted_text = path.read_text(encoding="utf-8")
            combined = f"{rec}\n{rows}\n{persisted_text}\n{stderr.getvalue()}"

            self.assertEqual(rec["error"], 'top API_TOKEN: [redacted] "api_key":[redacted] Authorization: [redacted]')
            self.assertEqual(rows[0]["tmux_stderr"], "stderr API_TOKEN: [redacted] AUTH: [redacted]")
            self.assertEqual(rows[0]["tmux_attempts"][0]["stderr"], "nested password: [redacted]")
            self.assertEqual(rows[0]["metadata"]["api_key"], "[redacted]")
            self.assertEqual(rows[0]["metadata"]["auth_header"], "[redacted]")
            self.assertIn('top API_TOKEN: [redacted] "api_key":[redacted] Authorization: [redacted]', stderr.getvalue())
            for secret in ("top-secret", "json-secret", "auth-secret-token", "stderr-secret", "QWxhZGRpbjpvcGVuIHNlc2FtZQ", "nested-secret", "meta-secret", "custom-secret"):
                self.assertNotIn(secret, combined)

    def test_failed_launch_redactor_handles_unclosed_quotes_and_colons(self) -> None:
        # redact_launch_failure_text is a pure util free function.
        examples = [
            ('API_TOKEN="secret-token', 'API_TOKEN=[redacted]'),
            ("PASSWORD='hunter2", 'PASSWORD=[redacted]'),
            ('API_TOKEN: secret-token', 'API_TOKEN: [redacted]'),
            ('password: hunter2', 'password: [redacted]'),
            ('"api_key":"json-secret"', '"api_key":[redacted]'),
            ('"password":"hunter2', '"password":[redacted]'),
            ('Authorization: Bearer abcdefghijklmnop', 'Authorization: [redacted]'),
            ('AUTH: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==', 'AUTH: [redacted]'),
            ('Authorization=Bearer abcdefghijklmnop', 'Authorization=[redacted]'),
            ('Authorization: [redacted] abcdefghijklmnop', 'Authorization: [redacted]'),
            ('API_TOKEN: [redacted]', 'API_TOKEN: [redacted]'),
            ('API_TOKEN=[redacted]', 'API_TOKEN=[redacted]'),
        ]

        for raw, expected in examples:
            with self.subTest(raw=raw):
                self.assertEqual(redact_launch_failure_text(raw), expected)

    def test_failed_launch_row_redacts_error_secrets(self) -> None:
        # launch_attempt_row is a pure launch_ledger free function; bind the same
        # constants server._launch_attempt_row binds.
        row = _launch_attempt_row_from_record(
            {
                "launch_id": "launch-secret-row",
                "state": "failed",
                "stage": "shell_startup",
                "error": "failed API_TOKEN: secret-token \"password\":\"hunter2\"",
                "agent_backend": "codex",
                "cwd": "/tmp/work",
                "created_ts": time.time(),
            }
        )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["launch_error"], "failed API_TOKEN: [redacted] \"password\":[redacted]")

    def test_list_sessions_exposes_pending_launch_as_session_row(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "launches.jsonl"
            store = _store(root)
            lock = threading.Lock()
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
                path=ledger,
            )

            coordinator = _list_coordinator(store=store, sessions={}, lock=lock, ledger_path=ledger)
            rows = coordinator.list_sessions()

        self.assertEqual(rows[0]["session_id"], "launch-pending")
        self.assertEqual(rows[0]["launch_state"], "tmux_pane_created")
        self.assertEqual(rows[0]["busy"], False)

    def test_delete_real_session_hides_stale_pending_launch_row(self) -> None:
        # Listing + lifecycle coordinators share the same store (and therefore
        # the same hidden_sessions set) and sessions dict. sock_call is the
        # injected socket boundary that the original mgr._sock_call stub stood
        # in for; no server module global is patched.
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "launches.jsonl"
            store = _store(root)
            lock = threading.Lock()
            now = time.time()
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
                path=ledger,
            )
            sessions = {
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
                    sock_path=root / "broker-live.sock",
                    transport="tmux",
                    launch_id="launch-live",
                    spawn_nonce="nonce-live",
                )
            }

            list_coordinator = _list_coordinator(store=store, sessions=sessions, lock=lock, ledger_path=ledger)
            lifecycle = _lifecycle_coordinator(
                store=store,
                sessions=sessions,
                lock=lock,
                ledger_path=ledger,
                sock_call=lambda _sock, _payload, *, timeout_s=1.0: {"ok": True},
            )

            before = list_coordinator.list_sessions()
            self.assertEqual([row["session_id"] for row in before], ["broker-live"])

            self.assertTrue(lifecycle.delete_session("broker-live"))
            after = list_coordinator.list_sessions()

        self.assertEqual(after, [])
        self.assertIn("launch-live", store.hidden_sessions)

    def test_prune_preserves_specific_existing_launch_failure(self) -> None:
        # SessionPruneCoordinator.prune_dead_sessions with latest_launch_attempt
        # bound to the temp ledger. The session socket does not exist, so it is
        # collected as dead; because the existing launch record is already
        # "failed", _record_pruned_launch_failure returns early and the original
        # failure record is preserved (not overwritten).
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "launches.jsonl"
            lock = threading.Lock()
            now = time.time()
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
                path=ledger,
            )
            sessions = {
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
                    sock_path=root / "missing.sock",
                    transport="tmux",
                    launch_id="launch-shell",
                )
            }

            prune = SessionPruneCoordinator(
                lock=lock,
                sessions=lambda: sessions,
                sock_call=lambda *a, **kw: {"ok": True},
                broker_busy_queue_from_state=lambda _state: (False, 0),
                broker_interrupted_idle_from_state=lambda _state: False,
                sock_error_definitely_stale=lambda _exc: True,
                pid_alive=lambda _pid: False,
                latest_launch_attempt=lambda launch_id: latest_launch_attempt(launch_id, path=ledger),
                submitted_user_messages=submitted_user_messages,
                launch_failure_tail=launch_failure_tail,
                which_tmux=lambda _name: "/usr/bin/tmux",
                tmux_pane_snapshot=lambda *a, **kw: {},
                clean_optional_text=server._clean_optional_text,
                record_launch_attempt=lambda rec: record_launch_attempt(rec, path=ledger, stderr=io.StringIO()),
                clear_deleted_session_state=lambda _sid: None,
                unlink_quiet=lambda _p: None,
            )

            prune.prune_dead_sessions()
            rows = read_launch_attempts(path=ledger, max_records=10, max_age_s=3600)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], "shell_startup")

    def test_prune_records_post_log_bound_recovery_for_incomplete_dead_session(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "launches.jsonl"
            lock = threading.Lock()
            now = time.time()
            log_path = root / "post-log.jsonl"
            log_path.write_text(
                '{"type":"event_msg","ts":1.0,"payload":{"type":"user_message","message":"POST_LOG_BOUND_DEATH_SENTINEL"}}\n',
                encoding="utf-8",
            )
            append_launch_attempt(
                {
                    "launch_id": "launch-post-log-prune",
                    "state": "log_bound",
                    "agent_backend": "codex",
                    "cwd": "/tmp/work",
                    "created_ts": now,
                    "updated_ts": now,
                    "log_path": str(log_path),
                },
                path=ledger,
            )
            sessions = {
                "broker-dead": Session(
                    session_id="broker-dead",
                    thread_id="rollout-thread",
                    broker_pid=1234,
                    codex_pid=1235,
                    agent_backend="codex",
                    owned=True,
                    start_ts=now,
                    cwd="/tmp/work",
                    log_path=log_path,
                    sock_path=root / "missing.sock",
                    launch_id="launch-post-log-prune",
                )
            }
            prune = SessionPruneCoordinator(
                lock=lock,
                sessions=lambda: sessions,
                sock_call=lambda *a, **kw: {"ok": True},
                broker_busy_queue_from_state=lambda _state: (False, 0),
                broker_interrupted_idle_from_state=lambda _state: False,
                sock_error_definitely_stale=lambda _exc: True,
                pid_alive=lambda _pid: False,
                latest_launch_attempt=lambda launch_id: latest_launch_attempt(launch_id, path=ledger),
                submitted_user_messages=submitted_user_messages,
                launch_failure_tail=launch_failure_tail,
                which_tmux=lambda _name: None,
                tmux_pane_snapshot=lambda *a, **kw: {},
                clean_optional_text=server._clean_optional_text,
                record_launch_attempt=lambda rec: record_launch_attempt(rec, path=ledger, stderr=io.StringIO()),
                clear_deleted_session_state=lambda _sid: None,
                unlink_quiet=lambda _p: None,
            )

            prune.prune_dead_sessions()
            rows = read_launch_attempts(path=ledger, max_records=10, max_age_s=3600)
            row = _launch_attempt_row_from_record(rows[0])
            payload = launch_attempt_transcript_payload(rows[0])

        self.assertEqual(rows[0]["state"], "failed")
        self.assertEqual(rows[0]["stage"], "session_pruned_after_log_bind")
        self.assertEqual(rows[0]["session_id"], "broker-dead")
        self.assertEqual(rows[0]["thread_id"], "rollout-thread")
        assert row is not None
        self.assertEqual(row["session_id"], "broker-dead")
        self.assertEqual(row["busy"], False)
        self.assertEqual([ev["role"] for ev in payload["events"]], ["user", "assistant"])
        self.assertEqual(payload["events"][0]["text"], "POST_LOG_BOUND_DEATH_SENTINEL")
        self.assertEqual(payload["events"][1]["message_class"], "error")

    def test_prune_carries_prelog_user_message_and_tmux_tail(self) -> None:
        # The tmux binary lookup (which_tmux) and pane capture
        # (tmux_pane_snapshot) are genuine OS/process boundaries; they are
        # injected as coordinator fields (returning a fixed binary path and a
        # fixed pane tail) instead of patching codoxear.server.shutil.which /
        # codoxear.server._tmux_pane_snapshot. The pre-log user message is
        # carried from the existing ledger record and the tmux pane tail from
        # the injected snapshot into the newly recorded failure.
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "launches.jsonl"
            lock = threading.Lock()
            now = time.time()
            append_launch_attempt(
                {
                    "launch_id": "launch-prelog",
                    "state": "broker_meta_bound",
                    "agent_backend": "codex",
                    "cwd": "/tmp/work",
                    "transport": "tmux",
                    "tmux_session": "codoxear",
                    "tmux_window": "work-dead",
                    "tmux_pane_id": "%9",
                    "created_ts": now,
                    "updated_ts": now,
                    "submitted_user_messages": [{"text": "copy me", "ts": now + 0.1, "source": "send"}],
                },
                path=ledger,
            )
            sessions = {
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
                    sock_path=root / "missing.sock",
                    transport="tmux",
                    tmux_session="codoxear",
                    tmux_window="work-dead",
                    launch_id="launch-prelog",
                )
            }

            def fake_tmux_pane_snapshot(_bin, *, pane_id=None, window=None):
                return {"tmux_pane_tail": "backend died\n", "tmux_pane_dead_status": "1"}

            prune = SessionPruneCoordinator(
                lock=lock,
                sessions=lambda: sessions,
                sock_call=lambda *a, **kw: {"ok": True},
                broker_busy_queue_from_state=lambda _state: (False, 0),
                broker_interrupted_idle_from_state=lambda _state: False,
                sock_error_definitely_stale=lambda _exc: True,
                pid_alive=lambda _pid: False,
                latest_launch_attempt=lambda launch_id: latest_launch_attempt(launch_id, path=ledger),
                submitted_user_messages=submitted_user_messages,
                launch_failure_tail=launch_failure_tail,
                which_tmux=lambda _name: "/usr/bin/tmux",
                tmux_pane_snapshot=fake_tmux_pane_snapshot,
                clean_optional_text=server._clean_optional_text,
                record_launch_attempt=lambda rec: record_launch_attempt(rec, path=ledger, stderr=io.StringIO()),
                clear_deleted_session_state=lambda _sid: None,
                unlink_quiet=lambda _p: None,
            )

            prune.prune_dead_sessions()
            rows = read_launch_attempts(path=ledger, max_records=10, max_age_s=3600)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "failed")
        self.assertEqual(rows[0]["submitted_user_messages"][0]["text"], "copy me")
        self.assertEqual(rows[0]["tmux_pane_tail"], "backend died\n")

    def test_list_sessions_omits_successful_launch_attempt_without_active_session(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "launches.jsonl"
            store = _store(root)
            lock = threading.Lock()
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
                path=ledger,
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
                path=ledger,
            )

            coordinator = _list_coordinator(store=store, sessions={}, lock=lock, ledger_path=ledger)
            rows = coordinator.list_sessions()

        self.assertEqual(rows, [])

    def test_delete_session_dismisses_launch_attempt_row(self) -> None:
        # With no live session, delete_session falls through to the launch-row
        # scan and hides the matching launch id. The listing coordinator then
        # filters it out via the shared hidden_sessions set.
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = root / "launches.jsonl"
            store = _store(root)
            lock = threading.Lock()
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
                path=ledger,
            )
            launch_id = str(rec["launch_id"])

            list_coordinator = _list_coordinator(store=store, sessions={}, lock=lock, ledger_path=ledger)
            lifecycle = _lifecycle_coordinator(
                store=store,
                sessions={},
                lock=lock,
                ledger_path=ledger,
                sock_call=lambda *a, **kw: {"ok": True},
            )

            self.assertTrue(lifecycle.delete_session(launch_id))
            rows = list_coordinator.list_sessions()

        self.assertEqual(rows, [])
        self.assertIn(launch_id, store.hidden_sessions)


if __name__ == "__main__":
    unittest.main()
