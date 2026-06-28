import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER_PY = ROOT / "codoxear" / "broker.py"
SERVER_PY = ROOT / "codoxear" / "server.py"
SESSION_CONTROL_PY = ROOT / "codoxear" / "session_control.py"
CONTROL_ROUTES_PY = ROOT / "codoxear" / "control_routes.py"


class TestInterruptSemanticsSource(unittest.TestCase):
    def test_web_interrupt_marks_broker_interrupt_request(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        control_runtime_source = SESSION_CONTROL_PY.read_text(encoding="utf-8")
        control_source = CONTROL_ROUTES_PY.read_text(encoding="utf-8")

        self.assertIn('resp = manager.inject_keys(session_id, "\\\\x1b", interrupt=True)', control_source)
        self.assertIn('_handle_control_post_route(', server_source)
        self.assertIn('def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False, interrupt: bool = False)', server_source)
        self.assertIn('if interrupt:\n                request["interrupt"] = True', control_runtime_source)
        self.assertIn('mark_interrupt = req.get("interrupt") is True and b == b"\\x1b"', broker_source)
        self.assertIn('_mark_explicit_interrupt_request(st, _now())', broker_source)
        self.assertIn('"interrupted_idle": (not st.busy) and st.last_interrupted_idle_ts > 0.0', broker_source)

    def test_explicit_interrupt_is_the_only_no_candidate_idle_relaxation(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        clear_start = broker_source.index('def _should_clear_busy_state')
        clear_end = broker_source.index('def _mark_explicit_interrupt_request', clear_start)
        block = broker_source[clear_start:clear_end]

        self.assertIn('if st.turn_open and (not st.turn_has_completion_candidate):', block)
        self.assertIn('if st.last_interrupt_request_ts <= 0.0:\n            return False', block)
        self.assertIn('BUSY_INTERRUPT_GRACE_SECONDS', block)
        self.assertIn('if st.pending_calls:\n        return False', block)


if __name__ == "__main__":
    unittest.main()
