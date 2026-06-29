import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
BROKER_TURN_STATE_PY = ROOT / "codoxear" / "broker_turn_state.py"
BROKER_CONTROL_PY = ROOT / "codoxear" / "broker_control.py"
SERVER_PY = ROOT / "codoxear" / "server.py"
SERVER_HANDLER_PY = ROOT / "codoxear" / "server_handler.py"
SESSION_CONTROL_PY = ROOT / "codoxear" / "session_control.py"
CONTROL_ROUTES_PY = ROOT / "codoxear" / "control_routes.py"


class TestInterruptSemanticsSource(unittest.TestCase):
    def test_web_interrupt_marks_broker_interrupt_request(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        handler_source = SERVER_HANDLER_PY.read_text(encoding="utf-8")
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        broker_control_source = BROKER_CONTROL_PY.read_text(encoding="utf-8")
        control_runtime_source = SESSION_CONTROL_PY.read_text(encoding="utf-8")
        control_source = CONTROL_ROUTES_PY.read_text(encoding="utf-8")

        self.assertIn('resp = manager.inject_keys(session_id, "\\\\x1b", interrupt=True)', control_source)
        self.assertIn('if handle_control_post_route(', handler_source)
        self.assertIn('def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False, interrupt: bool = False)', server_source)
        self.assertIn('if interrupt:\n                request["interrupt"] = True', control_runtime_source)
        self.assertIn('from codoxear.broker_control import _handle_broker_control_connection', broker_source)
        self.assertIn('mark_interrupt = req.get("interrupt") is True and b == b"\\x1b"', broker_control_source)
        self.assertIn('_mark_explicit_interrupt_request(st, now())', broker_control_source)
        self.assertIn('"interrupted_idle": (not st.busy) and st.last_interrupted_idle_ts > 0.0', broker_control_source)

    def test_explicit_interrupt_is_the_only_no_candidate_idle_relaxation(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        turn_state_source = BROKER_TURN_STATE_PY.read_text(encoding="utf-8")
        clear_start = turn_state_source.index('def _should_clear_busy_state')
        clear_end = turn_state_source.index('def _mark_explicit_interrupt_request', clear_start)
        block = turn_state_source[clear_start:clear_end]

        self.assertIn('from codoxear.broker_turn_state import _should_clear_busy_state as _should_clear_busy_state_impl', broker_source)
        self.assertIn('busy_quiet_seconds=BUSY_QUIET_SECONDS', broker_source)
        self.assertIn('busy_interrupt_grace_seconds=BUSY_INTERRUPT_GRACE_SECONDS', broker_source)
        self.assertIn('if st.turn_open and (not st.turn_has_completion_candidate):', block)
        self.assertIn('if st.last_interrupt_request_ts <= 0.0:\n            return False', block)
        self.assertIn('busy_interrupt_grace_seconds', block)
        self.assertIn('if st.pending_calls:\n        return False', block)


if __name__ == "__main__":
    unittest.main()
