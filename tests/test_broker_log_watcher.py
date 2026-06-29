import unittest
from pathlib import Path

from codoxear.broker_log_watcher import _clear_resume_delivery_mute_if_idle
from codoxear.broker_log_watcher import _pop_key_queue_if_idle
from codoxear.broker_turn_state import State


def _state() -> State:
    return State(
        codex_pid=1,
        pty_master_fd=9,
        cwd="/tmp",
        start_ts=0.0,
        codex_home=Path("/tmp"),
        sessions_dir=Path("/tmp"),
    )


class TestBrokerLogWatcherHelpers(unittest.TestCase):
    def test_pop_key_queue_only_when_idle(self) -> None:
        st = _state()
        st.key_queue = [b"a", b"b"]
        fd, queue = _pop_key_queue_if_idle(st)
        self.assertEqual(fd, 9)
        self.assertEqual(queue, [b"a", b"b"])
        self.assertEqual(st.key_queue, [])

        st.key_queue = [b"c"]
        st.busy = True
        fd, queue = _pop_key_queue_if_idle(st)
        self.assertIsNone(fd)
        self.assertEqual(queue, [])
        self.assertEqual(st.key_queue, [b"c"])

    def test_clear_resume_delivery_mute_only_when_idle(self) -> None:
        st = _state()
        st.resume_session_id = "resume-a"
        st.turn_open = True
        self.assertFalse(_clear_resume_delivery_mute_if_idle(st))
        self.assertEqual(st.resume_session_id, "resume-a")

        st.turn_open = False
        self.assertTrue(_clear_resume_delivery_mute_if_idle(st))
        self.assertIsNone(st.resume_session_id)


if __name__ == "__main__":
    unittest.main()
