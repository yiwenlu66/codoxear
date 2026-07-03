import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from codoxear.sessiond import Sessiond
from codoxear.sessiond import State
from codoxear.sessiond import _busy_value_after_log_batch
from codoxear.sessiond import _log_busy_signals
from codoxear.sessiond import _read_jsonl_from_offset
from codoxear.sessiond_control import SessiondControlDeps
from codoxear.sessiond_control import handle_sessiond_control_connection


def _sessiond_state(*, codex_pid: int, sock_path: Path) -> State:
    return State(
        session_id="sid",
        codex_pid=codex_pid,
        log_path=Path("/tmp/pending.jsonl"),
        sock_path=sock_path,
        pty_master_fd=1,
        start_ts=0.0,
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


class TestSessiondFailClosed(unittest.TestCase):
    def test_log_busy_signals_treat_pi_aborted_as_turn_end(self) -> None:
        self.assertEqual(
            _log_busy_signals({"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "aborted"}}),
            (False, True),
        )
        self.assertEqual(
            _log_busy_signals({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}}),
            (True, False),
        )

    def test_log_busy_batch_preserves_record_order(self) -> None:
        aborted = {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "aborted"}}
        user = {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "new turn"}]}}
        self.assertIs(_busy_value_after_log_batch([aborted, user]), True)
        self.assertIs(_busy_value_after_log_batch([user, aborted]), False)

    def test_log_busy_signals_cover_claude_code_user_and_final_assistant_rows(self) -> None:
        cc_user = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "run"}]}}
        cc_final = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}}
        self.assertEqual(_log_busy_signals(cc_user), (True, False))
        self.assertEqual(_log_busy_signals(cc_final), (False, True))
        self.assertIs(_busy_value_after_log_batch([cc_user, cc_final]), False)

    def test_sessiond_control_state_uses_runtime_state_schema(self) -> None:
        captured: dict[str, object] = {}

        def fake_control_connection(_conn, *, handlers, **_kwargs):
            response, after = handlers["state"]({})
            captured["response"] = response
            captured["after"] = after

        state = _sessiond_state(codex_pid=1234, sock_path=Path("/tmp/sessiond.sock"))
        state.busy = True
        deps = SessiondControlDeps(
            lock=threading.Lock(),
            state=lambda: state,
            encode_enter=lambda: b"\r",
            seq_bytes=lambda raw: raw.encode("utf-8"),
            write_all=lambda _fd, _data: None,
            inject=lambda **_kwargs: None,
            teardown_managed_process_group=lambda: None,
            handle_control_socket_connection=fake_control_connection,
            send_json_line=lambda _conn, _obj: None,
            socket_peer_disconnected=lambda _exc: False,
            print_exception=lambda: None,
        )

        handle_sessiond_control_connection(object(), deps=deps)

        self.assertEqual(captured["response"], {"busy": True, "queue_len": 0, "interrupted_idle": False})
        self.assertIsNone(captured["after"])

    def test_sessiond_reader_preserves_missing_file_contract(self) -> None:
        missing = Path(tempfile.gettempdir()) / "codoxear-missing-sessiond-jsonl-for-test.jsonl"
        try:
            missing.unlink()
        except FileNotFoundError:
            pass

        self.assertEqual(_read_jsonl_from_offset(missing, 17, max_bytes=4096), ([], 17))

    def test_teardown_managed_process_group_kills_real_process_group(self) -> None:
        proc = subprocess.Popen(["sh", "-c", "sleep 100"], start_new_session=True)
        try:
            sessiond = Sessiond(cwd="/tmp", codex_args=[])
            sessiond.state = _sessiond_state(codex_pid=proc.pid, sock_path=Path("/tmp/test-sessiond.sock"))

            sessiond._teardown_managed_process_group(wait_seconds=0.2)

            proc.wait(timeout=2.0)
            self.assertIsNotNone(proc.returncode)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)

    def test_sock_server_accept_failure_triggers_teardown(self) -> None:
        sessiond = Sessiond(cwd="/tmp", codex_args=[])
        with tempfile.TemporaryDirectory() as td:
            sessiond.state = _sessiond_state(codex_pid=1234, sock_path=Path(td) / "sessiond.sock")
            with patch("codoxear.sessiond.socket.socket", return_value=_AcceptCrashSocket()):
                with patch("codoxear.sessiond.os.chmod"):
                    with patch.object(sessiond, "_teardown_managed_process_group") as teardown:
                        sessiond._sock_server()

        teardown.assert_called_once_with()

    def test_discover_log_binds_pi_open_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pi_log = root / "pi-session.jsonl"
            pi_log.write_text('{"type":"session","id":"pi-sid","cwd":"/repo"}\n', encoding="utf-8")
            pending = root / "pending.jsonl"
            pending.write_text("", encoding="utf-8")
            sock_path = root / "sessiond.sock"
            sessiond = Sessiond(cwd="/repo", codex_args=[])
            sessiond.state = _sessiond_state(codex_pid=1234, sock_path=sock_path)
            sessiond.state.log_path = pending

            with patch("codoxear.sessiond.AGENT_BACKEND", "pi"):
                with patch("codoxear.sessiond.PENDING_DIR", root):
                    with patch("codoxear.sessiond._proc_find_open_rollout_log", return_value=pi_log) as find_log:
                        with patch("codoxear.sessiond._read_session_meta_payload", return_value={"id": "pi-sid", "cwd": "/repo"}):
                            with patch.object(sessiond, "_write_meta") as write_meta:
                                sessiond._discover_log()

            find_log.assert_called_once()
            self.assertEqual(find_log.call_args.kwargs["agent_backend"], "pi")
            self.assertEqual(find_log.call_args.kwargs["cwd"], "/repo")
            self.assertEqual(sessiond.state.session_id, "pi-sid")
            self.assertEqual(sessiond.state.log_path, pi_log)
            self.assertFalse(pending.exists())
            write_meta.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
