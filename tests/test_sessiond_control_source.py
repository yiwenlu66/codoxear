import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SESSIOND = ROOT / "codoxear" / "sessiond.py"
SESSIOND_CONTROL = ROOT / "codoxear" / "sessiond_control.py"


class TestSessiondControlSource(unittest.TestCase):
    def test_sessiond_control_handlers_have_dedicated_owner(self) -> None:
        sessiond_source = SESSIOND.read_text(encoding="utf-8")
        control_source = SESSIOND_CONTROL.read_text(encoding="utf-8")

        self.assertIn("from .sessiond_control import SessiondControlDeps", sessiond_source)
        self.assertIn("from .sessiond_control import handle_sessiond_control_connection", sessiond_source)
        self.assertIn("def _handle_conn(self, conn: socket.socket) -> None:", sessiond_source)
        self.assertIn("handle_sessiond_control_connection(", sessiond_source)
        self.assertIn("inject=_inject", sessiond_source)
        self.assertIn("encode_enter=_encode_enter", sessiond_source)
        self.assertIn("seq_bytes=_seq_bytes", sessiond_source)
        self.assertIn("write_all=_write_all", sessiond_source)
        self.assertIn("print_exception=traceback.print_exc", sessiond_source)
        self.assertIn("handle_control_socket_connection=_handle_control_socket_connection", sessiond_source)
        self.assertNotIn("def state_handler(", sessiond_source)
        self.assertNotIn("def send_handler(", sessiond_source)
        self.assertNotIn("def keys_handler(", sessiond_source)
        self.assertNotIn("def shutdown_handler(", sessiond_source)

        self.assertIn("class SessiondControlDeps:", control_source)
        self.assertIn("def handle_sessiond_control_connection(", control_source)
        self.assertIn("def state_handler(", control_source)
        self.assertIn("def tail_handler(", control_source)
        self.assertIn("def send_handler(", control_source)
        self.assertIn("def keys_handler(", control_source)
        self.assertIn("def shutdown_handler(", control_source)
        self.assertIn('{"error": "text required"}', control_source)
        self.assertIn('return {"error": "no state"}, None', control_source)
        self.assertIn('return {"error": "no pty", "commit_unknown": False}, None', control_source)
        self.assertIn('return {"error": str(e), "commit_unknown": True}, None', control_source)
        self.assertIn('return {"queued": False, "queue_len": 0}, None', control_source)
        self.assertIn('return {"ok": True}, deps.teardown_managed_process_group', control_source)


if __name__ == "__main__":
    unittest.main()
