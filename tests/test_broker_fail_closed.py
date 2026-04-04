import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from codoxear.broker import Broker
from codoxear.broker import State
from codoxear.agent_backend import get_agent_backend


def _broker_state(*, codex_pid: int, sock_path: Path) -> State:
    return State(
        codex_pid=codex_pid,
        pty_master_fd=1,
        cwd="/tmp",
        start_ts=0.0,
        codex_home=Path("/tmp"),
        sessions_dir=Path("/tmp"),
        sock_path=sock_path,
    )


class _AcceptCrashSocket:
    def bind(self, _addr: str) -> None:
        return

    def listen(self, _backlog: int) -> None:
        return

    def settimeout(self, _timeout: float) -> None:
        return

    def accept(self) -> tuple[object, object]:
        raise RuntimeError("boom")

    def close(self) -> None:
        return


class _FakeThread:
    started_targets: list[str] = []

    def __init__(self, *, target, daemon: bool) -> None:
        self._target = target
        self._daemon = daemon

    def start(self) -> None:
        self.started_targets.append(self._target.__name__)


class TestBrokerFailClosed(unittest.TestCase):
    def test_pi_broker_injects_explicit_session_path_for_new_sessions(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        with tempfile.TemporaryDirectory() as td, patch("codoxear.broker.sys.stdin", fake_stdin), patch.dict(
            "os.environ", {"PI_HOME": td, "CODEX_WEB_RESUME_SESSION_ID": ""}, clear=False
        ), patch("codoxear.broker.AGENT_BACKEND", "pi"), patch("codoxear.broker.BACKEND", get_agent_backend("pi")):
            broker = Broker(cwd="/tmp/pi-work", codex_args=["--model", "gpt-5.4"])

        self.assertEqual(broker.codex_args[:2], ["--model", "gpt-5.4"])
        self.assertEqual(broker.codex_args[-2], "--session")
        session_path = Path(broker.codex_args[-1])
        self.assertTrue(str(session_path).startswith(str(Path(td) / "agent" / "sessions" / "--tmp-pi-work--")))
        self.assertIsNone(broker._resume_session_id)

    def test_pi_discover_log_watcher_switches_when_new_log_appears_while_current_exists(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        with tempfile.TemporaryDirectory() as td, patch("codoxear.broker.sys.stdin", fake_stdin), patch.dict(
            "os.environ", {"PI_HOME": td}, clear=False
        ), patch("codoxear.broker.AGENT_BACKEND", "pi"), patch("codoxear.broker.BACKEND", get_agent_backend("pi")):
            broker = Broker(cwd="/tmp", codex_args=[])
            current = Path(td) / "current.jsonl"
            current.write_text('{"type":"session","id":"current","cwd":"/tmp"}\n', encoding="utf-8")
            new = Path(td) / "new.jsonl"
            new.write_text('{"type":"session","id":"new","cwd":"/tmp"}\n', encoding="utf-8")
            broker.state = _broker_state(codex_pid=1234, sock_path=Path(td) / "broker.sock")
            broker.state.log_path = current
            broker.state.known_rollout_paths = {current}
            broker.state.sock_path = Path(td) / "broker.sock"
            seen: list[Path] = []

            def _capture_switch(*, log_path: Path) -> None:
                seen.append(log_path)
                broker._stop.set()

            with patch("codoxear.broker._proc_find_open_rollout_log", return_value=new), patch.object(
                broker, "_maybe_register_or_switch_rollout", side_effect=_capture_switch
            ), patch("codoxear.broker.time.sleep", side_effect=lambda _seconds: None):
                broker._discover_log_watcher()

        self.assertEqual(seen, [new])

    def test_teardown_managed_process_group_kills_real_process_group(self) -> None:
        proc = subprocess.Popen(["sh", "-c", "sleep 100"], start_new_session=True)
        try:
            broker = Broker(cwd="/tmp", codex_args=[])
            broker.state = _broker_state(codex_pid=proc.pid, sock_path=Path("/tmp/test-broker.sock"))

            broker._teardown_managed_process_group(wait_seconds=0.2)

            proc.wait(timeout=2.0)
            self.assertIsNotNone(proc.returncode)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)

    def test_discover_log_watcher_failure_triggers_teardown(self) -> None:
        broker = Broker(cwd="/tmp", codex_args=[])
        broker.state = _broker_state(codex_pid=1234, sock_path=Path("/tmp/test-broker.sock"))

        with patch("codoxear.broker._proc_find_open_rollout_log", side_effect=RuntimeError("boom")):
            with patch.object(broker, "_teardown_managed_process_group") as teardown:
                broker._discover_log_watcher()

        teardown.assert_called_once_with()

    def test_discover_log_watcher_switches_rollout_after_resume(self) -> None:
        broker = Broker(cwd="/repo", codex_args=[])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_log = root / "rollout-2026-03-30T10-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            new_log = root / "rollout-2026-03-30T10-05-00-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            old_log.write_text('{"type":"session_meta","payload":{"id":"thread-old","cwd":"/repo","source":"cli"}}\n', encoding="utf-8")
            new_log.write_text('{"type":"session_meta","payload":{"id":"thread-new","cwd":"/repo","source":"cli"}}\n', encoding="utf-8")

            broker.state = _broker_state(codex_pid=1234, sock_path=root / "broker.sock")
            broker.state.log_path = old_log

            with patch("codoxear.broker._proc_find_open_rollout_log", return_value=new_log) as find_log, patch.object(
                broker, "_maybe_register_or_switch_rollout"
            ) as switch_log, patch("codoxear.broker.time.sleep", side_effect=lambda _secs: broker._stop.set()):
                broker._discover_log_watcher()

        find_log.assert_called_once()
        self.assertEqual(find_log.call_args.kwargs["ignored_paths"], {old_log})
        switch_log.assert_called_once_with(log_path=new_log)

    def test_discover_log_watcher_does_not_flip_back_to_old_rollout(self) -> None:
        broker = Broker(cwd="/repo", codex_args=[])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_log = root / "rollout-2026-03-30T10-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            new_log = root / "rollout-2026-03-30T10-05-00-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
            old_log.write_text('{"type":"session_meta","payload":{"id":"thread-old","cwd":"/repo","source":"cli"}}\n', encoding="utf-8")
            new_log.write_text('{"type":"session_meta","payload":{"id":"thread-new","cwd":"/repo","source":"cli"}}\n', encoding="utf-8")

            broker.sessions_dir = root
            broker.state = _broker_state(codex_pid=1234, sock_path=root / "broker.sock")
            broker.state.log_path = old_log
            broker.state.last_rollout_path = old_log

            seen_ignored_paths: list[set[Path]] = []

            def _fake_find_open_rollout_log(**kwargs):
                ignored = set(kwargs["ignored_paths"])
                seen_ignored_paths.append(ignored)
                if len(seen_ignored_paths) == 1:
                    return new_log
                broker._stop.set()
                return None

            with patch("codoxear.broker._proc_find_open_rollout_log", side_effect=_fake_find_open_rollout_log), patch(
                "codoxear.broker.time.sleep", return_value=None
            ), patch("codoxear.broker.os.waitpid", return_value=(0, 0)), patch.object(broker, "_write_meta"):
                broker._discover_log_watcher()

        self.assertEqual({p.resolve() for p in seen_ignored_paths[0]}, {old_log.resolve()})
        self.assertEqual({p.resolve() for p in seen_ignored_paths[1]}, {old_log.resolve(), new_log.resolve()})

    def test_sock_server_accept_failure_triggers_teardown(self) -> None:
        broker = Broker(cwd="/tmp", codex_args=[])
        with tempfile.TemporaryDirectory() as td:
            sock_root = Path(td)
            broker.state = _broker_state(codex_pid=1234, sock_path=sock_root / "broker.sock")
            with patch("codoxear.broker.SOCK_DIR", sock_root):
                with patch("codoxear.broker.socket.socket", return_value=_AcceptCrashSocket()):
                    with patch("codoxear.broker.os.chmod"):
                        with patch.object(broker, "_teardown_managed_process_group") as teardown:
                            broker._sock_server()

        teardown.assert_called_once_with()

    def test_web_owned_broker_forwards_local_stdin_when_tty_is_present(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: True, fileno=lambda: 9)
        _FakeThread.started_targets = []
        with tempfile.TemporaryDirectory() as td:
            broker = Broker(cwd=td, codex_args=[])
            broker.sessions_dir = Path(td) / "sessions"
            with patch("codoxear.broker.OWNER_TAG", "web"), patch("codoxear.broker.sys.stdin", fake_stdin), patch(
                "codoxear.broker._require_proc"
            ), patch("codoxear.broker._term_size", return_value=(24, 80)), patch(
                "codoxear.broker.pty.fork", return_value=(1234, 55)
            ), patch("codoxear.broker._set_winsize"), patch(
                "codoxear.broker.signal.signal"
            ), patch("codoxear.broker.termios.tcgetattr", return_value=["saved"]), patch(
                "codoxear.broker.termios.tcsetattr"
            ), patch("codoxear.broker.tty.setraw"), patch(
                "codoxear.broker.os.waitpid", return_value=(1234, 0)
            ), patch(
                "codoxear.broker.os.close"
            ), patch.object(
                broker, "_write_meta"
            ), patch(
                "codoxear.broker.threading.Thread", _FakeThread
            ):
                broker._emulate_terminal = False
                exit_code = broker.run()

        self.assertEqual(exit_code, 0)
        self.assertIn("_stdin_to_pty", _FakeThread.started_targets)

    def test_web_owned_headless_broker_keeps_local_stdin_disabled_without_tty(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        _FakeThread.started_targets = []
        with tempfile.TemporaryDirectory() as td:
            with patch("codoxear.broker.sys.stdin", fake_stdin):
                broker = Broker(cwd=td, codex_args=[])
            broker.sessions_dir = Path(td) / "sessions"
            with patch("codoxear.broker.OWNER_TAG", "web"), patch("codoxear.broker.sys.stdin", fake_stdin), patch(
                "codoxear.broker._require_proc"
            ), patch("codoxear.broker._term_size", return_value=(24, 80)), patch(
                "codoxear.broker.pty.fork", return_value=(1234, 55)
            ), patch("codoxear.broker._set_winsize"), patch(
                "codoxear.broker.signal.signal"
            ), patch(
                "codoxear.broker.os.waitpid", return_value=(1234, 0)
            ), patch(
                "codoxear.broker.os.close"
            ), patch.object(
                broker, "_write_meta"
            ), patch(
                "codoxear.broker.threading.Thread", _FakeThread
            ):
                exit_code = broker.run()

        self.assertEqual(exit_code, 0)
        self.assertNotIn("_stdin_to_pty", _FakeThread.started_targets)

    def test_write_meta_tracks_resume_session_id(self) -> None:
        broker = Broker(cwd="/tmp", codex_args=["resume", "resume-a"])
        with tempfile.TemporaryDirectory() as td:
            sock_path = Path(td) / "broker.sock"
            log_path = Path(td) / "rollout-2026-03-29T10-00-00-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
            log_path.write_text("", encoding="utf-8")
            broker.state = _broker_state(codex_pid=1234, sock_path=sock_path)
            broker.state.session_id = "broker-1"
            broker.state.cwd = td
            broker.state.log_path = log_path
            broker.state.resume_session_id = "resume-a"

            broker._write_meta()
            meta_path = sock_path.with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["resume_session_id"], "resume-a")

            broker.state.resume_session_id = None
            broker._write_meta()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertIsNone(meta["resume_session_id"])

    def test_write_meta_uses_agent_backend_for_backend_field(self) -> None:
        with patch("codoxear.broker.AGENT_BACKEND", "pi"), patch(
            "codoxear.broker.BACKEND", get_agent_backend("pi")
        ):
            broker = Broker(cwd="/tmp", codex_args=[])
            with tempfile.TemporaryDirectory() as td:
                sock_path = Path(td) / "broker.sock"
                broker.state = _broker_state(codex_pid=1234, sock_path=sock_path)
                broker.state.session_id = "pi-broker-1"

                broker._write_meta()

                meta = json.loads(sock_path.with_suffix(".json").read_text(encoding="utf-8"))

        self.assertEqual(meta["backend"], "pi")
        self.assertEqual(meta["agent_backend"], "pi")

    def test_pi_resume_run_binds_log_path_from_session_arg_before_first_write(self) -> None:
        fake_stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 9)
        captured: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as td, patch("codoxear.broker.sys.stdin", fake_stdin), patch("codoxear.broker.AGENT_BACKEND", "pi"), patch(
            "codoxear.broker.BACKEND", get_agent_backend("pi")
        ):
            log_path = Path(td) / "pi-resume.jsonl"
            log_path.write_text('{"type":"session","id":"resume-a","cwd":"/tmp"}\n', encoding="utf-8")
            expected_size = log_path.stat().st_size
            broker = Broker(cwd="/tmp", codex_args=["--session", str(log_path)])
            broker.sessions_dir = log_path.parent

            def _capture_write_meta() -> None:
                st = broker.state
                captured["session_id"] = st.session_id if st else None
                captured["log_path"] = str(st.log_path) if st and st.log_path else None
                captured["log_off"] = st.log_off if st else None

            with patch("codoxear.broker._require_proc"), patch("codoxear.broker._term_size", return_value=(24, 80)), patch(
                "codoxear.broker.pty.fork", return_value=(1234, 55)
            ), patch("codoxear.broker._set_winsize"), patch(
                "codoxear.broker.signal.signal"
            ), patch(
                "codoxear.broker.os.waitpid", return_value=(1234, 0)
            ), patch(
                "codoxear.broker.os.close"
            ), patch.object(
                broker, "_write_meta", side_effect=_capture_write_meta
            ), patch(
                "codoxear.broker.threading.Thread", _FakeThread
            ):
                broker.run()

        self.assertEqual(captured["session_id"], "resume-a")
        self.assertEqual(captured["log_path"], str(log_path))
        self.assertEqual(captured["log_off"], expected_size)


if __name__ == "__main__":
    unittest.main()
